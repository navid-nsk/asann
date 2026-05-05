"""
Phase D — Tier-1 Tabular Multi-Seed Cross-Reproducibility Study
================================================================

Runs 28 OpenML regression datasets at 5 seeds each (= 140 runs).
Mirrors the molecular-tier methodology to give a defensible
cross-tier reproducibility story.

OpenML dataset IDs come directly from the user-provided table
(image showing OpenML task ID and dataset ID columns).

Usage:
    python tier1_openml_multiseed.py                    # all 28 x 5 seeds
    python tier1_openml_multiseed.py --datasets abalone tecator
    python tier1_openml_multiseed.py --seeds 42         # single seed (smoke)
    python tier1_openml_multiseed.py --max-epochs 50    # short runs

Output: phase_d_results/<dataset_name>/seed_<N>/phase_d_summary.json
plus all standard ASANN training artifacts (architecture_evolution.csv etc.)
"""
import sys
import os
import json
import time
import argparse
import traceback
from pathlib import Path

_exp_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(_exp_dir)
if _exp_dir not in sys.path:
    sys.path.insert(0, _exp_dir)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)


# ============================================================================
# OpenML registry — taken from the user's screenshot.
# Keys: short canonical name. Values: openml_did (dataset id, parens column).
# ============================================================================
OPENML_REGISTRY = {
    "moneyball":                            41021,
    "yprop_4_1":                              416,
    "sat11_hand_runtime_regression":        41980,
    "topo_2_1":                               422,
    "house_prices_nominal":                 42563,
    "mercedes_benz_greener_manufacturing":  42570,
    "abalone":                              42726,
    "colleges":                             42727,
    "us_crime":                             42730,
    "mip_2016_regression":                  43071,
    "airfoil_self_noise":                   44957,
    "auction_verification":                 44958,
    "concrete_compressive_strength":        44959,
    "energy_efficiency":                    44960,
    "geographical_origin_of_music":         44965,
    "student_performance_por":              44967,
    "qsar_fish_toxicity":                   44970,
    "grid_stability":                       44973,
    "cpu_activity":                         44978,
    "kin8nm":                               44980,
    "pumadyn32nh":                          44981,
    "cars":                                 44994,
    "tecator":                                505,
    "space_ga":                               507,
    "boston":                                 531,
    "socmob":                                 541,
    "sensory":                                546,
    "quake":                                  550,
}


SEEDS = (42, 137, 256, 314, 529)


def run_one(dataset_name: str, openml_did: int, seed: int,
            results_root: Path, max_epochs: int = None):
    """Run a single (dataset, seed) experiment and save summary."""
    out_dir = results_root / dataset_name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "phase_d_summary.json"

    # Skip-if-OK behaviour
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                rec = json.load(f)
            if rec.get("status") == "OK":
                print(f"  [SKIP] {dataset_name} seed={seed} (already OK)")
                return rec
        except Exception:
            pass  # treat as missing if unreadable

    print("=" * 80)
    print(f"  Phase D — {dataset_name} seed={seed} (openml_did={openml_did})")
    print("=" * 80)

    # Import the generic runner from tier_1
    from tier_1.exp_1_openml_generic import run_experiment_openml

    t0 = time.time()
    rec = {"dataset_name": dataset_name, "openml_did": openml_did, "seed": seed}
    try:
        metrics, arch, cfg_dict, train_metrics = run_experiment_openml(
            results_dir=str(out_dir),
            dataset_name=dataset_name,
            openml_did=openml_did,
            seed=seed,
            max_epochs_override=max_epochs,
        )
        rec.update({
            "status": "OK",
            "elapsed_seconds": time.time() - t0,
            "metrics": metrics,
            "arch": arch,
            "config": cfg_dict,
            "max_epochs": max_epochs,
        })
    except Exception as e:
        elapsed = time.time() - t0
        traceback.print_exc()
        rec.update({
            "status": f"ERROR: {type(e).__name__}: {e}",
            "elapsed_seconds": elapsed,
            "max_epochs": max_epochs,
        })

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, default=str)
    print(f"  Saved: {summary_path}")
    print(f"  Elapsed: {rec['elapsed_seconds']:.1f}s\n")
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Subset of dataset names; default = all")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Subset of seeds; default = all 5")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--results-dir", default="phase_d_results")
    args = parser.parse_args()

    datasets = args.datasets or list(OPENML_REGISTRY.keys())
    seeds = args.seeds or SEEDS

    # Validate
    unknown = [d for d in datasets if d not in OPENML_REGISTRY]
    if unknown:
        print(f"ERROR: unknown datasets {unknown}")
        print(f"Known: {list(OPENML_REGISTRY.keys())}")
        sys.exit(2)

    results_root = (Path(_proj_root) / "experiments" / args.results_dir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    print(f"Phase D: {len(datasets)} datasets x {len(seeds)} seeds = "
          f"{len(datasets) * len(seeds)} runs")
    print(f"Output: {results_root}")

    n_ok = n_err = n_skip = 0
    for ds in datasets:
        did = OPENML_REGISTRY[ds]
        for seed in seeds:
            rec = run_one(ds, did, seed, results_root, max_epochs=args.max_epochs)
            status = rec.get("status", "?")
            if status == "OK":
                if "(already OK)" in str(rec):
                    n_skip += 1
                else:
                    n_ok += 1
            else:
                n_err += 1

    print()
    print("=" * 80)
    print(f"Phase D summary: {n_ok} OK, {n_err} ERROR, {n_skip} SKIP")
    print("=" * 80)


if __name__ == "__main__":
    main()
