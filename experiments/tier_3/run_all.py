"""
ASANN Tier 3 -- Run All Image Classification Experiments
============================================================

Sequentially runs all 8 experiments:
  3a. MNIST          (28x28, 70K samples, 10 classes)
  3b. Fashion-MNIST  (28x28, 70K samples, 10 classes)
  3c. KMNIST         (28x28, 70K samples, 10 classes)
  3d. SVHN           (32x32x3, 99K samples, 10 classes)
  3e. CIFAR-10       (32x32x3, 60K samples, 10 classes)
  3f. CIFAR-100      (32x32x3, 60K samples, 100 classes)
  3g. STL-10         (96x96x3, 13K samples, 10 classes)
  3h. ImageNet16-120 (16x16x3, 158K samples, 120 classes)

Produces a combined summary table and saves combined_results.json.

Usage:
    python run_all.py
    python run_all.py --skip 3f 3g    # skip specific experiments
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
from common import setup_paths, run_classification_experiment_wrapper

# Import experiment modules
from exp_3a_mnist import run_experiment as run_3a
from exp_3b_fashion_mnist import run_experiment as run_3b
from exp_3c_kmnist import run_experiment as run_3c
from exp_3d_svhn import run_experiment as run_3d
from exp_3e_cifar10 import run_experiment as run_3e
from exp_3f_cifar100 import run_experiment as run_3f
from exp_3g_stl10 import run_experiment as run_3g
from exp_3h_imagenet16 import run_experiment as run_3h


EXPERIMENTS = [
    ("3a", "MNIST", run_3a),
    ("3b", "Fashion-MNIST", run_3b),
    ("3c", "KMNIST", run_3c),
    ("3d", "SVHN", run_3d),
    ("3e", "CIFAR-10", run_3e),
    ("3f", "CIFAR-100", run_3f),
    ("3g", "STL-10", run_3g),
    ("3h", "ImageNet16-120", run_3h),
]


def print_summary_table(results):
    """Print a formatted summary table of all classification results."""
    print("\n")
    print("=" * 130)
    print("  ASANN Tier 3 -- Combined Image Classification Results Summary")
    print("=" * 130)

    header = (
        f"{'Experiment':<18} | {'Accuracy':>8} | {'F1-macro':>8} | {'F1-wt':>8} | "
        f"{'Prec':>6} | {'Recall':>6} | {'Layers':>6} | {'Params':>10} | {'Time':>8} | {'Status':<10}"
    )
    print(header)
    print("-" * 130)

    for name, metrics, arch, elapsed, status in results:
        if metrics is not None:
            print(
                f"{name:<18} | "
                f"{metrics['accuracy']:>8.4f} | "
                f"{metrics['f1_macro']:>8.4f} | "
                f"{metrics['f1_weighted']:>8.4f} | "
                f"{metrics['precision_macro']:>6.4f} | "
                f"{metrics['recall_macro']:>6.4f} | "
                f"{arch['num_layers']:>6} | "
                f"{arch['total_parameters']:>10,} | "
                f"{elapsed:>7.1f}s | "
                f"{status:<10}"
            )
        else:
            print(
                f"{name:<18} | "
                f"{'---':>8} | "
                f"{'---':>8} | "
                f"{'---':>8} | "
                f"{'---':>6} | "
                f"{'---':>6} | "
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
    parser = argparse.ArgumentParser(description="Run all ASANN Tier 3 experiments")
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="Experiment IDs to skip (e.g., --skip 3f 3g)"
    )
    args = parser.parse_args()

    skip_set = set(args.skip)

    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_3"
    tier_results.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 60)
    print("  ASANN Tier 3 -- Image Classification Experiments")
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

        result = run_classification_experiment_wrapper(name, run_fn, tier_results)
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
