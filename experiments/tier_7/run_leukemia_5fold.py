"""Run Munich Leukemia Lab 5-fold cross-validation and aggregate per-class metrics.

Matches the evaluation protocol of Matek et al. (Nature Machine Intelligence 2019):
- 5-fold stratified cross-validation (80/20 per fold)
- Per-class Precision and Sensitivity (recall), mean +/- std across folds
"""
import subprocess
import sys
import os
import json
import numpy as np
from pathlib import Path

N_FOLDS = 5

proj_root = Path(__file__).resolve().parent.parent.parent
results_base = proj_root / "experiments" / "results" / "tier_7"
exp_script = proj_root / "experiments" / "tier_7" / "exp_7e_leukemia.py"

# Load class names
sys.path.insert(0, str(proj_root / "experiments"))
from tier_7.bio_utils import LEUKEMIA_CLASSES

all_results = {}

for fold in range(N_FOLDS):
    print(f"\n{'='*70}")
    print(f"  Leukemia 5-fold CV -- Fold {fold}/{N_FOLDS}")
    print(f"{'='*70}\n")

    env = os.environ.copy()
    env["LEUKEMIA_FOLD"] = str(fold)

    # Clean old results for this fold
    results_dir = results_base / f"leukemia_5fold_f{fold}"
    if results_dir.exists():
        import shutil
        shutil.rmtree(results_dir, ignore_errors=True)

    result = subprocess.run(
        [sys.executable, str(exp_script)],
        env=env,
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"  FAILED: fold {fold}")
        continue

    # Load results
    results_file = results_dir / f"results_fold{fold}.json"
    if results_file.exists():
        with open(results_file) as f:
            metrics = json.load(f)
        all_results[fold] = metrics
        acc = metrics.get("accuracy", "?")
        f1w = metrics.get("f1_weighted", "?")
        auroc = metrics.get("auroc_macro", "?")
        print(f"\n  Fold {fold}: Accuracy={acc:.4f}, F1w={f1w:.4f}, "
              f"AUROC={auroc:.4f}")
    else:
        print(f"  Results file not found: {results_file}")

# ===== Aggregate across folds =====
print(f"\n{'='*70}")
print(f"  5-FOLD CV SUMMARY ({len(all_results)}/{N_FOLDS} folds)")
print(f"{'='*70}")

if all_results:
    # Overall metrics
    accuracies = [r["accuracy"] for r in all_results.values()]
    bal_accs = [r.get("balanced_accuracy", 0) for r in all_results.values()]
    f1_macros = [r.get("f1_macro", 0) for r in all_results.values()]
    f1_weighteds = [r.get("f1_weighted", 0) for r in all_results.values()]
    aurocs = [r.get("auroc_macro", 0) for r in all_results.values()
              if r.get("auroc_macro") is not None]

    print(f"\n  Overall Metrics:")
    print(f"    Accuracy:          {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
    print(f"    Balanced Accuracy: {np.mean(bal_accs):.4f} +/- {np.std(bal_accs):.4f}")
    print(f"    F1 (macro):        {np.mean(f1_macros):.4f} +/- {np.std(f1_macros):.4f}")
    print(f"    F1 (weighted):     {np.mean(f1_weighteds):.4f} +/- {np.std(f1_weighteds):.4f}")
    if aurocs:
        print(f"    AUROC (macro):     {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")

    # Per-class Precision and Sensitivity (matching Matek et al. Table 1 format)
    print(f"\n  Per-class Precision and Sensitivity (5-fold CV):")
    print(f"  {'Class':30s}  {'Precision':>14s}  {'Sensitivity':>14s}  {'N_images':>10s}")
    print(f"  {'-'*74}")

    per_class_summary = {}
    for cls_name in LEUKEMIA_CLASSES:
        precs = []
        sens = []
        n_total = 0
        for fold, r in all_results.items():
            pc_prec = r.get("per_class_precision", {})
            pc_sens = r.get("per_class_sensitivity", {})
            pc_n = r.get("per_class_n_test", {})
            if cls_name in pc_prec:
                precs.append(pc_prec[cls_name])
            if cls_name in pc_sens:
                sens.append(pc_sens[cls_name])
            if cls_name in pc_n:
                n_total += pc_n[cls_name]

        if precs and sens:
            prec_mean = np.mean(precs)
            prec_std = np.std(precs)
            sens_mean = np.mean(sens)
            sens_std = np.std(sens)
            # n_total across all folds = each image appears in exactly 1 test fold
            print(f"  {cls_name:30s}  {prec_mean:.2f} +/- {prec_std:.2f}  "
                  f"{sens_mean:.2f} +/- {sens_std:.2f}  {n_total:>10d}")
            per_class_summary[cls_name] = {
                "precision_mean": float(prec_mean),
                "precision_std": float(prec_std),
                "sensitivity_mean": float(sens_mean),
                "sensitivity_std": float(sens_std),
                "n_images": int(n_total),
            }

    # Save aggregate
    agg_path = results_base / "leukemia_5fold_summary.json"
    agg = {
        "n_folds": len(all_results),
        "folds": list(all_results.keys()),
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "balanced_accuracy_mean": float(np.mean(bal_accs)),
        "balanced_accuracy_std": float(np.std(bal_accs)),
        "f1_macro_mean": float(np.mean(f1_macros)),
        "f1_macro_std": float(np.std(f1_macros)),
        "f1_weighted_mean": float(np.mean(f1_weighteds)),
        "f1_weighted_std": float(np.std(f1_weighteds)),
        "auroc_macro_mean": float(np.mean(aurocs)) if aurocs else None,
        "auroc_macro_std": float(np.std(aurocs)) if aurocs else None,
        "per_class": per_class_summary,
        "per_fold": {str(k): v for k, v in all_results.items()},
    }
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n  Summary saved to {agg_path}")
