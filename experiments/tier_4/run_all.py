"""
ASANN Tier 4 -- Run All Sequence / Time-Series Experiments
============================================================

Sequentially runs all 3 experiments:
  4b. Airline Passengers     — Univariate forecasting (144 monthly points)
  4c. ECG5000                — Time-series classification (5K samples, 140 steps)
  4d. ETTh1                  — Multivariate forecasting (17K hourly readings)

Produces a combined summary table and saves combined_results.json.

Usage:
    python run_all.py
    python run_all.py --skip 4a    # skip specific experiments
"""

import sys
import os
import json
import time
import argparse

# Add experiments/ dir (for common.py) and project root (for asann)
_exp_dir = os.path.join(os.path.dirname(__file__), "..")
_proj_root = os.path.join(os.path.dirname(__file__), "..", "..")
if _exp_dir not in sys.path:
    sys.path.insert(0, _exp_dir)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from pathlib import Path
from common import (
    setup_paths, run_classification_experiment_wrapper,
    run_forecasting_experiment_wrapper,
)

# Import experiment modules
from exp_4b_airline import run_experiment as run_4b
from exp_4c_ecg5000 import run_experiment as run_4c
from exp_4d_etth1 import run_experiment as run_4d


EXPERIMENTS = [
    ("4b", "Airline Passengers", run_4b, "forecasting"),
    ("4c", "ECG5000 Classification", run_4c, "classification"),
    ("4d", "ETTh1 Forecasting", run_4d, "forecasting"),
]


def print_summary_table(results):
    """Print a formatted summary table of all Tier 4 results."""
    print("\n")
    print("=" * 110)
    print("  ASANN Tier 4 -- Sequence / Time-Series Results Summary")
    print("=" * 110)

    header = (
        f"{'Experiment':<25} | {'Metric1':>10} | {'Metric2':>10} | "
        f"{'Layers':>6} | {'Params':>8} | {'Time':>7} | {'Status':<10}"
    )
    print(header)
    print("-" * 110)

    for name, metrics, arch, elapsed, status in results:
        if metrics is not None:
            # Pick the two most relevant metrics per experiment
            if 'perplexity' in metrics:
                m1_name, m1_val = "PPL", f"{metrics['perplexity']:.1f}"
                m2_name, m2_val = "Acc", f"{metrics.get('token_accuracy', 0):.4f}"
            elif 'accuracy' in metrics:
                m1_name, m1_val = "Acc", f"{metrics['accuracy']:.4f}"
                m2_name, m2_val = "F1", f"{metrics.get('f1_macro', 0):.4f}"
            else:
                m1_name, m1_val = "MAE", f"{metrics['mae']:.4f}"
                m2_name, m2_val = "RMSE", f"{metrics['rmse']:.4f}"

            print(
                f"{name:<25} | "
                f"{m1_name}={m1_val:>5} | "
                f"{m2_name}={m2_val:>5} | "
                f"{arch['num_layers']:>6} | "
                f"{arch['total_parameters']:>8} | "
                f"{elapsed:>6.1f}s | "
                f"{status:<10}"
            )
        else:
            print(
                f"{name:<25} | "
                f"{'---':>10} | "
                f"{'---':>10} | "
                f"{'---':>6} | "
                f"{'---':>8} | "
                f"{elapsed:>6.1f}s | "
                f"{status:<10}"
            )

    print("=" * 110)


def save_combined_results(results, results_dir):
    """Save combined results to JSON."""
    combined = {}
    for name, metrics, arch, elapsed, status in results:
        entry = {
            "status": status,
            "elapsed_seconds": elapsed,
        }
        if metrics is not None:
            entry["metrics"] = metrics
            entry["architecture"] = {
                "num_layers": arch["num_layers"],
                "total_parameters": arch["total_parameters"],
                "architecture_cost": arch["architecture_cost"],
                "widths": [l["out_features"] for l in arch["layers"]],
                "num_connections": len(arch.get("connections", [])),
            }
        combined[name] = entry

    path = os.path.join(results_dir, "combined_results.json")
    with open(path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Run all ASANN Tier 4 experiments")
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="Experiment IDs to skip (e.g., --skip 4a)"
    )
    args = parser.parse_args()

    skip_set = set(args.skip)

    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_4"
    tier_results.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 60)
    print("  ASANN Tier 4 -- Sequence / Time-Series Experiments")
    print("#" * 60)
    print(f"\n  Results directory: {tier_results}")
    print(f"  Experiments to run: {len(EXPERIMENTS) - len(skip_set)}")
    if skip_set:
        print(f"  Skipping: {skip_set}")

    total_start = time.time()
    results = []

    for exp_id, name, run_fn, task_type in EXPERIMENTS:
        if exp_id in skip_set:
            print(f"\n  Skipping {exp_id}: {name}")
            continue

        if task_type == "classification":
            result = run_classification_experiment_wrapper(name, run_fn, tier_results)
        else:
            result = run_forecasting_experiment_wrapper(name, run_fn, tier_results)
        results.append(result)

    total_elapsed = time.time() - total_start

    # Print summary
    print_summary_table(results)
    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # Save combined results
    save_combined_results(results, str(tier_results))

    # Report failures
    failures = [r for r in results if r[4] != "OK"]
    if failures:
        print(f"\n  WARNING: {len(failures)} experiment(s) failed:")
        for name, _, _, _, status in failures:
            print(f"    - {name}: {status}")
        return 1

    print("\n  All experiments completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
