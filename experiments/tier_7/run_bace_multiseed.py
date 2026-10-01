"""Run BACE scaffold experiment with multiple seeds and aggregate results."""
import subprocess
import sys
import os
import json
import numpy as np
from pathlib import Path

SEEDS = [42, 137, 256, 314, 529]
SPLIT = "scaffold"

proj_root = Path(__file__).resolve().parent.parent.parent
results_base = proj_root / "experiments" / "results" / "tier_7"
exp_script = proj_root / "experiments" / "tier_7" / "exp_7i_moleculenet_bace.py"

all_results = {}

for seed in SEEDS:
    print(f"\n{'='*70}")
    print(f"  BACE {SPLIT} split -- seed {seed}")
    print(f"{'='*70}\n")

    env = os.environ.copy()
    env["BACE_SPLIT"] = SPLIT
    env["BACE_SEED"] = str(seed)

    # Clean old results for this seed
    results_dir = results_base / f"moleculenet_bace_{SPLIT}_s{seed}"
    if results_dir.exists():
        import shutil
        shutil.rmtree(results_dir, ignore_errors=True)

    result = subprocess.run(
        [sys.executable, str(exp_script)],
        env=env,
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"  FAILED: seed {seed}")
        continue

    # Load results
    results_file = results_dir / f"results_{SPLIT}.json"
    if results_file.exists():
        with open(results_file) as f:
            metrics = json.load(f)
        all_results[seed] = metrics
        print(f"\n  Seed {seed}: AUROC={metrics.get('auroc', '?'):.4f}, "
              f"KA-GNN buggy={metrics.get('auroc_buggy_max', '?'):.4f}, "
              f"FedLG buggy={metrics.get('auroc_fedlg_buggy_max', '?'):.4f}")
    else:
        print(f"  Results file not found: {results_file}")

# Summary
print(f"\n{'='*70}")
print(f"  MULTI-SEED SUMMARY ({len(all_results)}/{len(SEEDS)} seeds)")
print(f"{'='*70}")

if all_results:
    aurocs = [r["auroc"] for r in all_results.values()]
    buggy_maxs = [r.get("auroc_buggy_max", 0) for r in all_results.values()]
    fedlg_maxs = [r.get("auroc_fedlg_buggy_max", 0) for r in all_results.values()]

    print(f"\n  Correct AUROC:")
    for seed, r in sorted(all_results.items()):
        print(f"    seed {seed}: {r['auroc']:.4f}")
    print(f"    Mean: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")
    print(f"    Best: {np.max(aurocs):.4f}")

    print(f"\n  KA-GNN Buggy AUROC (max across best+final models, 128-sample subsets x 501 trials):")
    for seed, r in sorted(all_results.items()):
        bm = r.get('auroc_buggy_best_model', r.get('auroc_buggy_max', 0))
        bf = r.get('auroc_buggy_final_model', 0)
        print(f"    seed {seed}: {r.get('auroc_buggy_max', 0):.4f} "
              f"(best_model={bm:.4f}, final_model={bf:.4f})")
    print(f"    Mean: {np.mean(buggy_maxs):.4f} +/- {np.std(buggy_maxs):.4f}")
    print(f"    Best: {np.max(buggy_maxs):.4f}")

    print(f"\n  FedLG Buggy AUROC (per-batch averaging + max across epochs):")
    for seed, r in sorted(all_results.items()):
        fb = r.get('auroc_fedlg_buggy_best_model', 0)
        ff = r.get('auroc_fedlg_buggy_final_model', 0)
        print(f"    seed {seed}: {r.get('auroc_fedlg_buggy_max', 0):.4f} "
              f"(best_model={fb:.4f}, final_model={ff:.4f})")
    print(f"    Mean: {np.mean(fedlg_maxs):.4f} +/- {np.std(fedlg_maxs):.4f}")
    print(f"    Best: {np.max(fedlg_maxs):.4f}")

    # Save aggregate
    agg_path = results_base / f"bace_{SPLIT}_multiseed_summary.json"
    agg = {
        "seeds": list(all_results.keys()),
        "per_seed": {str(k): v for k, v in all_results.items()},
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "auroc_best": float(np.max(aurocs)),
        "kagnn_buggy_max_mean": float(np.mean(buggy_maxs)),
        "kagnn_buggy_max_std": float(np.std(buggy_maxs)),
        "kagnn_buggy_max_best": float(np.max(buggy_maxs)),
        "fedlg_buggy_max_mean": float(np.mean(fedlg_maxs)),
        "fedlg_buggy_max_std": float(np.std(fedlg_maxs)),
        "fedlg_buggy_max_best": float(np.max(fedlg_maxs)),
    }
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n  Summary saved to {agg_path}")
