"""Run ESOL scaffold experiment with multiple seeds and aggregate results."""
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
exp_script = proj_root / "experiments" / "tier_7" / "exp_7g_moleculenet_esol.py"

all_results = {}

for seed in SEEDS:
    print(f"\n{'='*70}")
    print(f"  ESOL {SPLIT} split -- seed {seed}")
    print(f"{'='*70}\n")

    env = os.environ.copy()
    env["ESOL_SPLIT"] = SPLIT
    env["ESOL_SEED"] = str(seed)

    # Clean old results for this seed
    results_dir = results_base / f"moleculenet_esol_{SPLIT}_s{seed}"
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
        print(f"\n  Seed {seed}: RMSE={metrics.get('rmse', '?'):.4f}, "
              f"FedLG buggy={metrics.get('rmse_fedlg_buggy_min', '?'):.4f}")
    else:
        print(f"  Results file not found: {results_file}")

# Summary
print(f"\n{'='*70}")
print(f"  MULTI-SEED SUMMARY ({len(all_results)}/{len(SEEDS)} seeds)")
print(f"{'='*70}")

if all_results:
    rmses = [r["rmse"] for r in all_results.values()]
    maes = [r.get("mae", 0) for r in all_results.values()]
    r2s = [r.get("r2", 0) for r in all_results.values()]
    fedlg_mins = [r.get("rmse_fedlg_buggy_min", 0) for r in all_results.values()]

    print(f"\n  Correct RMSE:")
    for seed, r in sorted(all_results.items()):
        print(f"    seed {seed}: {r['rmse']:.4f} "
              f"(MAE={r.get('mae', 0):.4f}, R2={r.get('r2', 0):.4f})")
    print(f"    Mean: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
    print(f"    Best: {np.min(rmses):.4f}")

    print(f"\n  FedLG Buggy RMSE (per-batch averaging + min across epochs):")
    for seed, r in sorted(all_results.items()):
        fb = r.get('rmse_fedlg_buggy_best_model', 0)
        ff = r.get('rmse_fedlg_buggy_final_model', 0)
        print(f"    seed {seed}: {r.get('rmse_fedlg_buggy_min', 0):.4f} "
              f"(best_model={fb:.4f}, final_model={ff:.4f})")
    print(f"    Mean: {np.mean(fedlg_mins):.4f} +/- {np.std(fedlg_mins):.4f}")
    print(f"    Best: {np.min(fedlg_mins):.4f}")

    # Save aggregate
    agg_path = results_base / f"esol_{SPLIT}_multiseed_summary.json"
    agg = {
        "seeds": list(all_results.keys()),
        "per_seed": {str(k): v for k, v in all_results.items()},
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "rmse_best": float(np.min(rmses)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "fedlg_buggy_min_mean": float(np.mean(fedlg_mins)),
        "fedlg_buggy_min_std": float(np.std(fedlg_mins)),
        "fedlg_buggy_min_best": float(np.min(fedlg_mins)),
    }
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n  Summary saved to {agg_path}")
