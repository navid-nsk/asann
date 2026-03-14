"""
ASANN Tier 1 — Run All Tabular Regression Experiments
=======================================================

Sequentially runs all 4 experiments:
  1a. California Housing  (sklearn, 20640 samples, 8 features)
  1c. Ames Housing        (OpenML, 1460 samples, 79+ features)
  1d. Bike Sharing        (OpenML, 17389 samples, 16 features)
  1e. Energy Efficiency   (OpenML, 768 samples, 8 features)

Produces a combined summary table and saves combined_results.json.

Usage:
    python run_all.py
    python run_all.py --skip 1c 1d    # skip specific experiments
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
from exp_1a_california import run_experiment as run_1a
from exp_1c_ames_housing import run_experiment as run_1c
from exp_1d_bike_sharing import run_experiment as run_1d
from exp_1e_energy_efficiency import run_experiment as run_1e


EXPERIMENTS = [
    ("1a", "California Housing", run_1a),
    ("1c", "Ames Housing", run_1c),
    ("1d", "Bike Sharing", run_1d),
    ("1e", "Energy Efficiency", run_1e),
]


def print_summary_table(results):
    """Print a formatted summary table of all experiment results."""
    print("\n")
    print("=" * 110)
    print("  ASANN Tier 1 — Combined Results Summary")
    print("=" * 110)

    header = (
        f"{'Experiment':<22} | {'MSE':>12} | {'RMSE':>10} | {'MAE':>10} | "
        f"{'R²':>8} | {'Layers':>6} | {'Params':>8} | {'Time':>7} | {'Status':<10}"
    )
    print(header)
    print("-" * 110)

    for name, metrics, arch, elapsed, status in results:
        if metrics is not None:
            widths = [l["out_features"] for l in arch["layers"]]
            print(
                f"{name:<22} | "
                f"{metrics['mse']:>12.4f} | "
                f"{metrics['rmse']:>10.4f} | "
                f"{metrics['mae']:>10.4f} | "
                f"{metrics['r2']:>8.4f} | "
                f"{arch['num_layers']:>6} | "
                f"{arch['total_parameters']:>8} | "
                f"{elapsed:>6.1f}s | "
                f"{status:<10}"
            )
        else:
            print(
                f"{name:<22} | "
                f"{'---':>12} | "
                f"{'---':>10} | "
                f"{'---':>10} | "
                f"{'---':>8} | "
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
    parser = argparse.ArgumentParser(description="Run all ASANN Tier 1 experiments")
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="Experiment IDs to skip (e.g., --skip 1c 1d)"
    )
    args = parser.parse_args()

    skip_set = set(args.skip)

    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_1"
    tier_results.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 60)
    print("  ASANN Tier 1 -- Tabular Regression Experiments")
    print("#" * 60)
    print(f"\n  Results directory: {tier_results}")
    print(f"  Experiments to run: {len(EXPERIMENTS) - len(skip_set)}")
    if skip_set:
        print(f"  Skipping: {skip_set}")

    total_start = time.time()
    results = []

    for exp_id, name, run_fn in EXPERIMENTS:
        if exp_id in skip_set:
            print(f"\n  Skipping {exp_id}: {name}")
            continue

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

    print("\n  All experiments completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
