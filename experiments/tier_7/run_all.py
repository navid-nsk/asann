"""
Tier 7: Run All Biological/Biomedical Experiments
==================================================
Runs all 8 Tier 7 experiments sequentially and prints summary.
Gracefully handles missing packages (SKIPPED status).
"""

import sys
import os
import time

_exp_dir = os.path.join(os.path.dirname(__file__), "..")
_proj_root = os.path.join(os.path.dirname(__file__), "..", "..")
if _exp_dir not in sys.path:
    sys.path.insert(0, _exp_dir)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from pathlib import Path
from common import setup_paths

# Import experiment runners
from tier_7.exp_7a_gdsc import run_experiment as run_7a
from tier_7.exp_7b_moleculenet import run_experiment as run_7b
from tier_7.exp_7c_tcga import run_experiment as run_7c
from tier_7.exp_7d_celltype import run_experiment as run_7d
from tier_7.exp_7e_leukemia import run_experiment as run_7e
from tier_7.exp_7f_moleculenet_hiv import run_experiment as run_7f
from tier_7.exp_7g_moleculenet_esol import run_experiment as run_7g
from tier_7.exp_7h_moleculenet_muv import run_experiment as run_7h


EXPERIMENTS = [
    ("7A", "GDSC Drug Sensitivity",      run_7a,  "regression"),
    ("7B", "MoleculeNet BBBP",           run_7b,  "classification"),
    ("7C", "TCGA Cancer Classification",  run_7c,  "classification"),
    ("7D", "Cell Type Classification",    run_7d,  "classification"),
    ("7E", "Leukemia Blood Cell",        run_7e,  "classification"),
    ("7F", "MoleculeNet HIV",            run_7f,  "classification"),
    ("7G", "MoleculeNet ESOL",           run_7g,  "regression"),
    ("7H", "MoleculeNet MUV",            run_7h,  "classification"),
]


def run_single(exp_id, name, run_fn, task_type, results_base):
    """Run a single experiment with error handling."""
    results_dir = results_base / name.lower().replace(" ", "_")
    results_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    try:
        result = run_fn(str(results_dir))
        elapsed = time.time() - start
        if result is not None:
            metrics = result[0] if len(result) >= 1 else {}
            return "OK", elapsed, metrics
        return "OK", elapsed, {}
    except RuntimeError as e:
        elapsed = time.time() - start
        msg = str(e)
        if msg.startswith("SKIPPED:"):
            return msg, elapsed, {}
        return f"FAILED: {msg}", elapsed, {}
    except Exception as e:
        elapsed = time.time() - start
        import traceback
        traceback.print_exc()
        return f"FAILED: {e}", elapsed, {}


def main():
    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_7"
    tier_results.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  TIER 7: Biological / Biomedical Dataset Experiments")
    print("=" * 70)

    results = []
    total_start = time.time()

    for exp_id, name, run_fn, task_type in EXPERIMENTS:
        print(f"\n{'='*70}")
        print(f"  [{exp_id}] {name}")
        print(f"{'='*70}")
        status, elapsed, metrics = run_single(
            exp_id, name, run_fn, task_type, tier_results
        )
        results.append((exp_id, name, status, elapsed, metrics))

        # Print quick status
        if status == "OK":
            key_metric = ""
            if task_type == "classification":
                acc = metrics.get("accuracy", metrics.get("test_accuracy", "?"))
                auroc = metrics.get("auroc", "")
                key_metric = f"acc={acc}"
                if auroc:
                    key_metric += f", auroc={auroc}"
            else:
                r2 = metrics.get("r2", "?")
                key_metric = f"R2={r2}"
            print(f"\n  -> {status} ({elapsed:.1f}s) {key_metric}")
        else:
            print(f"\n  -> {status} ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_start

    # ===== Summary Table =====
    print(f"\n\n{'='*70}")
    print("  TIER 7 SUMMARY")
    print(f"{'='*70}")
    print(f"  {'ID':<5} {'Experiment':<30} {'Status':<12} {'Time':>8}")
    print(f"  {'-'*5} {'-'*30} {'-'*12} {'-'*8}")

    ok_count = 0
    skip_count = 0
    fail_count = 0

    for exp_id, name, status, elapsed, metrics in results:
        if status == "OK":
            ok_count += 1
            status_short = "OK"
        elif "SKIPPED" in status:
            skip_count += 1
            status_short = "SKIPPED"
        else:
            fail_count += 1
            status_short = "FAILED"

        print(f"  {exp_id:<5} {name:<30} {status_short:<12} {elapsed:>7.1f}s")

    print(f"\n  Total: {ok_count} OK, {skip_count} SKIPPED, {fail_count} FAILED")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
