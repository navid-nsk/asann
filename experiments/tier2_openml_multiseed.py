"""
Phase E — Tier-2 Tabular Classification Multi-Seed Cross-Reproducibility
=========================================================================

Runs 29 OpenML classification datasets at 5 seeds each (= 145 runs).
OpenML dataset IDs come from the user-provided table (image with task ID
and dataset ID columns). Reports ROC AUC + balanced accuracy + F1.

Usage:
    python tier2_openml_multiseed.py
    python tier2_openml_multiseed.py --datasets phoneme yeast
    python tier2_openml_multiseed.py --seeds 42

Output: phase_e_results/<dataset_name>/seed_<N>/phase_e_summary.json
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
# OpenML registry — taken from user's screenshot.
# Keys: short canonical name. Values: openml_did (parens column).
# ============================================================================
OPENML_REGISTRY = {
    "pc4":                                  1049,
    "kc1":                                  1067,
    "mfeat_factors":                          12,
    "blood_transfusion_service_center":     1464,
    "first_order_theorem_proving":          1475,
    "ozone_level_8hr":                      1487,
    "phoneme":                              1489,
    "qsar_biodeg":                          1494,
    "yeast":                                 181,
    "eucalyptus":                            188,
    "cmc":                                    23,
    "credit_g":                               31,
    "kr_vs_kp":                                3,
    "wine_quality_white":                  40498,
    "dna":                                 40670,
    "churn":                               40701,
    "satellite":                           40900,
    "car":                                 40975,
    "australian":                          40981,
    "steel_plates_fault":                  40982,
    "wilt":                                40983,
    "segment":                             40984,
    "jasmine":                             41143,
    "madeline":                            41144,
    "philippine":                          41145,
    "sylvine":                             41146,
    "ada":                                 41156,
    "gesture_phase_segmentation_processed": 4538,
    "vehicle":                                54,
}


SEEDS = (42, 137, 256, 314, 529)


def run_one(dataset_name: str, openml_did: int, seed: int,
            results_root: Path, max_epochs: int = None):
    out_dir = results_root / dataset_name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "phase_e_summary.json"

    if summary_path.exists():
        try:
            with open(summary_path) as f:
                rec = json.load(f)
            if rec.get("status") == "OK":
                print(f"  [SKIP] {dataset_name} seed={seed} (already OK)")
                return rec
        except Exception:
            pass

    print("=" * 80)
    print(f"  Phase E — {dataset_name} seed={seed} (openml_did={openml_did})")
    print("=" * 80)

    from tier_2.exp_2_openml_generic import run_experiment_openml_classification

    t0 = time.time()
    rec = {"dataset_name": dataset_name, "openml_did": openml_did, "seed": seed}
    try:
        metrics, arch, cfg_dict, train_metrics = run_experiment_openml_classification(
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
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--results-dir", default="phase_e_results")
    args = parser.parse_args()

    datasets = args.datasets or list(OPENML_REGISTRY.keys())
    seeds = args.seeds or SEEDS

    unknown = [d for d in datasets if d not in OPENML_REGISTRY]
    if unknown:
        print(f"ERROR: unknown datasets {unknown}")
        print(f"Known: {list(OPENML_REGISTRY.keys())}")
        sys.exit(2)

    results_root = (Path(_proj_root) / "experiments" / args.results_dir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    print(f"Phase E: {len(datasets)} datasets x {len(seeds)} seeds = "
          f"{len(datasets) * len(seeds)} runs")
    print(f"Output: {results_root}")

    n_ok = n_err = n_skip = 0
    for ds in datasets:
        did = OPENML_REGISTRY[ds]
        for seed in seeds:
            rec = run_one(ds, did, seed, results_root, max_epochs=args.max_epochs)
            status = rec.get("status", "?")
            if status == "OK":
                n_ok += 1
            else:
                n_err += 1

    print()
    print("=" * 80)
    print(f"Phase E summary: {n_ok} OK, {n_err} ERROR")
    print("=" * 80)


if __name__ == "__main__":
    main()
