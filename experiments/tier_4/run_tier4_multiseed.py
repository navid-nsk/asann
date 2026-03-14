"""
Run all Tier 4 experiments with multiple seeds and aggregate results.
=====================================================================
Experiments:
  - Airline Passengers (log-transformed forecasting)
  - ECG5000 (time-series classification)
  - ETTh1 (multivariate forecasting, 96-step horizon)

Each experiment is run with 5 seeds. Results are aggregated per-experiment.
"""
import subprocess
import sys
import os
import json
import time
import shutil
import numpy as np
from pathlib import Path

SEEDS = [42, 137, 256, 314, 529]

proj_root = Path(__file__).resolve().parent.parent.parent
results_base = proj_root / "experiments" / "results" / "tier_4"
results_base.mkdir(parents=True, exist_ok=True)

tier4_dir = proj_root / "experiments" / "tier_4"


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "airline": {
        "script": tier4_dir / "exp_4b_airline.py",
        "seed_env": "AIRLINE_SEED",
        "results_prefix": "airline_passengers",
        "results_file": lambda seed: f"results_s{seed}.json",
        "key_metrics": ["rmse_log", "mape_original", "mae_log"],
        "primary_metric": "rmse_log",
        "primary_label": "RMSE (log)",
        "secondary_metrics": {
            "mape_original": "MAPE (%)",
            "mae_log": "MAE (log)",
        },
        "type": "forecasting",
        "sota": "SARIMA: RMSE ~0.09-0.10 (log), MAPE ~1.6-2.1%",
    },
    "ecg5000": {
        "script": tier4_dir / "exp_4c_ecg5000.py",
        "seed_env": "ECG5000_SEED",
        "results_prefix": "ecg5000_classification",
        "results_file": lambda seed: f"results_s{seed}.json",
        "key_metrics": ["accuracy", "f1_weighted", "auc_ovr", "balanced_accuracy"],
        "primary_metric": "accuracy",
        "primary_label": "Accuracy",
        "secondary_metrics": {
            "f1_weighted": "F1 (weighted)",
            "auc_ovr": "AUC (OvR)",
            "balanced_accuracy": "Balanced Acc",
        },
        "type": "classification",
        "sota": "InceptionTime/ResNet: 99.1% - 99.6%",
    },
    "etth1": {
        "script": tier4_dir / "exp_4d_etth1.py",
        "seed_env": "ETTH1_SEED",
        "results_prefix": "etth1_forecasting",
        "results_file": lambda seed: f"results_s{seed}.json",
        "key_metrics": ["mse_scaled", "mae_scaled"],
        "primary_metric": "mse_scaled",
        "primary_label": "MSE (scaled)",
        "secondary_metrics": {
            "mae_scaled": "MAE (scaled)",
        },
        "type": "forecasting",
        "sota": "PatchTST: MSE ~0.370, MAE ~0.390 (96-step horizon)",
    },
}


def run_single_experiment(exp_name: str, exp_config: dict, seed: int):
    """Run a single experiment with a given seed. Returns metrics dict or None."""
    print(f"\n{'='*70}")
    print(f"  {exp_name} -- seed {seed}")
    print(f"{'='*70}\n")

    env = os.environ.copy()
    env[exp_config["seed_env"]] = str(seed)

    # Clean old results for this seed
    results_dir = results_base / f"{exp_config['results_prefix']}_s{seed}"
    if results_dir.exists():
        shutil.rmtree(results_dir, ignore_errors=True)

    result = subprocess.run(
        [sys.executable, str(exp_config["script"])],
        env=env,
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"  FAILED: {exp_name} seed {seed}")
        return None

    # Load results
    results_file = results_dir / exp_config["results_file"](seed)
    if results_file.exists():
        with open(results_file) as f:
            metrics = json.load(f)
        pm = exp_config["primary_metric"]
        print(f"\n  Seed {seed}: {exp_config['primary_label']}="
              f"{metrics.get(pm, '?'):.4f}")
        return metrics
    else:
        # Try the common results.json as fallback
        alt_file = results_dir / "results.json"
        if alt_file.exists():
            with open(alt_file) as f:
                data = json.load(f)
            if "metrics" in data:
                return data["metrics"]
            return data
        print(f"  Results file not found: {results_file}")
        return None


def aggregate_results(exp_name: str, exp_config: dict,
                      all_results: dict) -> dict:
    """Aggregate results across seeds and print summary."""
    print(f"\n{'='*70}")
    print(f"  {exp_name.upper()} MULTI-SEED SUMMARY "
          f"({len(all_results)}/{len(SEEDS)} seeds)")
    print(f"{'='*70}")

    if not all_results:
        print("  No successful runs!")
        return {}

    pm = exp_config["primary_metric"]
    pm_label = exp_config["primary_label"]

    # Primary metric
    values = [r[pm] for r in all_results.values() if pm in r]
    print(f"\n  {pm_label}:")
    for seed, r in sorted(all_results.items()):
        extras = []
        for mk, ml in exp_config["secondary_metrics"].items():
            if mk in r:
                extras.append(f"{ml}={r[mk]:.4f}")
        extra_str = f" ({', '.join(extras)})" if extras else ""
        print(f"    seed {seed}: {r.get(pm, '?'):.4f}{extra_str}")
    if values:
        print(f"    Mean: {np.mean(values):.4f} +/- {np.std(values):.4f}")
        print(f"    Best: {np.min(values):.4f}")
    print(f"  SOTA reference: {exp_config['sota']}")

    # Build aggregate dict
    agg = {
        "experiment": exp_name,
        "seeds": list(all_results.keys()),
        "n_seeds": len(all_results),
        "per_seed": {str(k): v for k, v in all_results.items()},
    }

    for mk in exp_config["key_metrics"]:
        vals = [r[mk] for r in all_results.values() if mk in r]
        if vals:
            agg[f"{mk}_mean"] = float(np.mean(vals))
            agg[f"{mk}_std"] = float(np.std(vals))
            agg[f"{mk}_best"] = float(np.min(vals))

    return agg


def main():
    overall_start = time.time()

    # Determine which experiments to run (all by default, or specify via CLI)
    if len(sys.argv) > 1:
        exp_names = [a for a in sys.argv[1:] if a in EXPERIMENTS]
        if not exp_names:
            print(f"Usage: {sys.argv[0]} [airline] [ecg5000] [etth1]")
            print(f"Available: {list(EXPERIMENTS.keys())}")
            sys.exit(1)
    else:
        exp_names = list(EXPERIMENTS.keys())

    all_summaries = {}

    for exp_name in exp_names:
        exp_config = EXPERIMENTS[exp_name]
        exp_start = time.time()
        exp_results = {}

        for seed in SEEDS:
            metrics = run_single_experiment(exp_name, exp_config, seed)
            if metrics is not None:
                exp_results[seed] = metrics

        # Aggregate
        agg = aggregate_results(exp_name, exp_config, exp_results)
        if agg:
            agg["elapsed_seconds"] = time.time() - exp_start
            # Save per-experiment summary
            agg_path = results_base / f"{exp_name}_multiseed_summary.json"
            with open(agg_path, "w") as f:
                json.dump(agg, f, indent=2, default=str)
            print(f"\n  Summary saved to {agg_path}")
            all_summaries[exp_name] = agg

    # Final overview
    total_time = time.time() - overall_start
    print(f"\n\n{'#'*70}")
    print(f"  TIER 4 MULTI-SEED OVERVIEW")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'#'*70}\n")

    for exp_name, agg in all_summaries.items():
        exp_config = EXPERIMENTS[exp_name]
        pm = exp_config["primary_metric"]
        pm_label = exp_config["primary_label"]
        mean_val = agg.get(f"{pm}_mean", float("nan"))
        std_val = agg.get(f"{pm}_std", float("nan"))
        print(f"  {exp_name:12s}: {pm_label} = {mean_val:.4f} +/- {std_val:.4f} "
              f"({agg['n_seeds']}/{len(SEEDS)} seeds)")
        print(f"                 SOTA: {exp_config['sota']}")

    # Save overall summary
    overall_path = results_base / "tier4_multiseed_overview.json"
    with open(overall_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\n  Overall summary saved to {overall_path}")


if __name__ == "__main__":
    main()
