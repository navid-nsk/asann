"""
ASANN Tier 5 — Run All PDE Regression Experiments
====================================================

Sequentially runs all 6 PDE experiments using PINNacle benchmark data:
  5a. Poisson 2D         (1,255 samples, baseline elliptic)
  5b. Burgers 1D         (1,111 samples, shock formation)
  5d. Kuramoto-Sivashinsky (50k samples, chaotic 4th-order)
  5e. Poisson-Boltzmann 2D (3,236 samples, reaction-dominated)
  5f. Gray-Scott 2D      (50k samples, coupled reaction-diffusion)
  5g. Wave 2D Hetero     (50k samples, variable coefficients)

Produces a combined summary table and saves combined_results.json.

Usage:
    python run_all.py
    python run_all.py --skip 5c 5d    # skip specific experiments
    python run_all.py --only 5a 5b    # run only specific experiments
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
from tier_5.exp_5a_poisson2d import run_experiment as run_5a
from tier_5.exp_5b_burgers1d import run_experiment as run_5b
from tier_5.exp_5d_kuramoto_sivashinsky import run_experiment as run_5d
from tier_5.exp_5e_poisson_boltzmann2d import run_experiment as run_5e
from tier_5.exp_5f_grayscott import run_experiment as run_5f
from tier_5.exp_5g_wave2d_hetero import run_experiment as run_5g


EXPERIMENTS = [
    ("5a", "Poisson 2D", run_5a),
    ("5b", "Burgers 1D", run_5b),
    ("5d", "Kuramoto-Sivashinsky", run_5d),
    ("5e", "Poisson-Boltzmann 2D", run_5e),
    ("5f", "Gray-Scott 2D", run_5f),
    ("5g", "Wave 2D Hetero", run_5g),
]


def print_summary_table(results):
    """Print a formatted summary table of all PDE experiment results."""
    print("\n")
    print("=" * 130)
    print("  ASANN Tier 5 — PDE Regression Results Summary")
    print("=" * 130)

    header = (
        f"{'Experiment':<24} | {'MSE':>12} | {'RMSE':>10} | {'MAE':>10} | "
        f"{'R²':>8} | {'Layers':>6} | {'Params':>8} | "
        f"{'Deriv':>5} | {'Poly':>4} | {'Branch':>6} | "
        f"{'Time':>7} | {'Status':<10}"
    )
    print(header)
    print("-" * 130)

    for name, metrics, arch, elapsed, status in results:
        if metrics is not None:
            pde = metrics.get("pde", {})
            n_deriv = pde.get("derivative_ops_discovered", 0)
            n_poly = pde.get("polynomial_ops_discovered", 0)
            n_branch = pde.get("branched_ops_discovered", 0)
            print(
                f"{name:<24} | "
                f"{metrics['mse']:>12.6f} | "
                f"{metrics['rmse']:>10.6f} | "
                f"{metrics['mae']:>10.6f} | "
                f"{metrics['r2']:>8.4f} | "
                f"{arch['num_layers']:>6} | "
                f"{arch['total_parameters']:>8} | "
                f"{n_deriv:>5} | "
                f"{n_poly:>4} | "
                f"{n_branch:>6} | "
                f"{elapsed:>6.1f}s | "
                f"{status:<10}"
            )
        else:
            print(
                f"{name:<24} | "
                f"{'---':>12} | "
                f"{'---':>10} | "
                f"{'---':>10} | "
                f"{'---':>8} | "
                f"{'---':>6} | "
                f"{'---':>8} | "
                f"{'---':>5} | "
                f"{'---':>4} | "
                f"{'---':>6} | "
                f"{elapsed:>6.1f}s | "
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
            # Separate PDE metrics from regression metrics for clean JSON
            pde = metrics.pop("pde", {})
            entry["metrics"] = metrics
            entry["pde_discovery"] = {
                "derivative_ops": pde.get("derivative_ops_discovered", 0),
                "polynomial_ops": pde.get("polynomial_ops_discovered", 0),
                "branched_ops": pde.get("branched_ops_discovered", 0),
                "total_ops": pde.get("total_ops", 0),
                "ops_detail": pde.get("discovered_ops_detail", []),
                "architecture": pde.get("architecture", []),
            }
            if "treatment_counts" in pde:
                entry["pde_discovery"]["treatment_counts"] = pde["treatment_counts"]
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
    parser = argparse.ArgumentParser(description="Run all ASANN Tier 5 PDE experiments")
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="Experiment IDs to skip (e.g., --skip 5c 5d)"
    )
    parser.add_argument(
        "--only", nargs="*", default=[],
        help="Run only these experiment IDs (e.g., --only 5a 5b)"
    )
    args = parser.parse_args()

    skip_set = set(args.skip)
    only_set = set(args.only) if args.only else None

    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_5"
    tier_results.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 70)
    print("  ASANN Tier 5 -- PDE Regression Experiments (PINNacle Benchmark)")
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

    print("\n  All PDE experiments completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
