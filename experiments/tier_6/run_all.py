"""
ASANN Tier 6 -- Run All Graph Neural Network Experiments
==========================================================

Sequentially runs all 6 experiments:
  6a. CiteSeer      (3,327 nodes, citation graph, 6 classes)
  6b. PubMed        (19,717 nodes, citation graph, 3 classes)
  6c. OGBN-ArXiv    (169,343 nodes, citation graph, 40 classes)
  6d. Reddit        (232,965 nodes, social graph, 41 classes)
  6e. METR-LA       (traffic speed forecasting, 207 sensors)
  6f. PEMS-BAY      (traffic speed forecasting, 325 sensors)

Produces a combined summary table and saves combined_results.json.

Usage:
    python run_all.py
    python run_all.py --skip 6c 6d    # skip specific experiments
    python run_all.py --only 6a 6b    # run only specific experiments
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
from common import setup_paths, run_experiment_wrapper

# Import experiment modules
from tier_6.exp_6a_citeseer import run_experiment as run_6a
from tier_6.exp_6b_pubmed import run_experiment as run_6b
from tier_6.exp_6c_ogbn_arxiv import run_experiment as run_6c
from tier_6.exp_6d_reddit import run_experiment as run_6d
from tier_6.exp_6e_metr_la import run_experiment as run_6e
from tier_6.exp_6f_pems_bay import run_experiment as run_6f


EXPERIMENTS = [
    ("6a", "CiteSeer", run_6a),
    ("6b", "PubMed", run_6b),
    ("6c", "OGBN-ArXiv", run_6c),
    ("6d", "Reddit", run_6d),
    ("6e", "METR-LA", run_6e),
    ("6f", "PEMS-BAY", run_6f),
]


def print_summary_table(results):
    """Print a formatted summary table of all graph experiment results."""
    print("\n")
    print("=" * 130)
    print("  ASANN Tier 6 — Graph Neural Network Results Summary")
    print("=" * 130)

    header = (
        f"{'Experiment':<18} | {'Metric':>12} | {'Value':>10} | "
        f"{'Layers':>6} | {'Params':>10} | {'Time':>8} | {'Status':<10}"
    )
    print(header)
    print("-" * 130)

    for name, metrics, arch, elapsed, status in results:
        if metrics is not None:
            # Classification tasks show accuracy; forecasting tasks show MAE/RMSE
            if "accuracy" in metrics:
                metric_name = "Accuracy"
                metric_val = metrics["accuracy"]
            elif "test_accuracy" in metrics:
                metric_name = "Accuracy"
                metric_val = metrics["test_accuracy"]
            elif "mae" in metrics:
                metric_name = "MAE"
                metric_val = metrics["mae"]
            elif "rmse" in metrics:
                metric_name = "RMSE"
                metric_val = metrics["rmse"]
            elif "r2" in metrics:
                metric_name = "R²"
                metric_val = metrics["r2"]
            else:
                metric_name = "N/A"
                metric_val = 0.0

            print(
                f"{name:<18} | "
                f"{metric_name:>12} | "
                f"{metric_val:>10.4f} | "
                f"{arch['num_layers']:>6} | "
                f"{arch['total_parameters']:>10,} | "
                f"{elapsed:>7.1f}s | "
                f"{status:<10}"
            )
        else:
            print(
                f"{name:<18} | "
                f"{'---':>12} | "
                f"{'---':>10} | "
                f"{'---':>6} | "
                f"{'---':>10} | "
                f"{elapsed:>7.1f}s | "
                f"{status:<10}"
            )

    print("=" * 130)


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
    parser = argparse.ArgumentParser(description="Run all ASANN Tier 6 graph experiments")
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="Experiment IDs to skip (e.g., --skip 6c 6d)"
    )
    parser.add_argument(
        "--only", nargs="*", default=[],
        help="Run only these experiment IDs (e.g., --only 6a 6b)"
    )
    args = parser.parse_args()

    skip_set = set(args.skip)
    only_set = set(args.only) if args.only else None

    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_6"
    tier_results.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 70)
    print("  ASANN Tier 6 -- Graph Neural Network Experiments")
    print("#" * 70)
    print(f"\n  Results directory: {tier_results}")

    # Determine which experiments to run
    exps_to_run = []
    for exp_id, name, run_fn in EXPERIMENTS:
        if only_set is not None and exp_id not in only_set:
            continue
        if exp_id in skip_set:
            continue
        exps_to_run.append((exp_id, name, run_fn))

    print(f"  Experiments to run: {len(exps_to_run)}")
    if skip_set:
        print(f"  Skipping: {skip_set}")
    if only_set:
        print(f"  Only running: {only_set}")

    total_start = time.time()
    results = []

    for exp_id, name, run_fn in exps_to_run:
        result = run_experiment_wrapper(name, run_fn, tier_results)
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

    print("\n  All graph experiments completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
