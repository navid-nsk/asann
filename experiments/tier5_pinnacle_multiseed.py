"""
Phase F — Tier-5 PINNacle PDE Multi-Seed Cross-Reproducibility
==============================================================

Runs PDE benchmarks at 5 seeds each at high epoch budget (default 1500)
to give every initial state enough time to converge to a stable
architecture.

Targets:
  Existing PDEs needing multi-seed (only single-seed in Phase C):
    - kuramoto_sivashinsky (1D, 4th-order, quadratic advection)
    - grayscott (2D, cubic kinetics)
    - wave_darcy / wave2d_hetero (2D heterogeneous, linear)
  New PDEs (vocabulary-compatible):
    - poisson_classic (2D linear elliptic, Laplacian-only)
    - heat_multiscale (2D parabolic, multi-scale diffusion)
    - heat_darcy (2D parabolic, Darcy-media variable coef)
    - heat_complex (2D parabolic, complex domain)
    - heat_longtime (2D parabolic, long-time horizon)

Vocabulary check (all required ops are in surgery.py CANDIDATE_OPERATIONS
or SPATIAL_CANDIDATE_OPERATIONS):
  - 2D Laplacian: derivative_conv2d_laplacian ✓
  - 2D 1st-order: derivative_conv2d_dx, dy ✓
  - 2D 2nd-order: derivative_conv2d_dxx, dyy ✓
  - 2D variable-coef diffusion: spatial_branched_diff_react ✓
  - 1D 1st/2nd-order: derivative_conv1d_o1, _o2 ✓
  - polynomial nonlinearity: polynomial_deg2, _deg3 ✓
  - spatial polynomial: spatial_polynomial_deg2, _deg3 ✓
  - branched deriv+poly: branched_deriv_poly ✓
  - temporal diff: temporal_diff ✓

So none of the new PDEs require ops outside the existing vocabulary.

Usage:
    python tier5_pinnacle_multiseed.py
    python tier5_pinnacle_multiseed.py --pdes heat_multiscale poisson_classic
    python tier5_pinnacle_multiseed.py --max-epochs 1500
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


# Datasets to run multi-seed in Phase F.
# Existing PDEs that previously had only 1 seed in Phase C-1.
EXISTING_PDES = [
    "kuramoto_sivashinsky",   # 1D KS chaos
    "grayscott",              # 2D cubic reaction-diffusion
    "wave_darcy",             # 2D heterogeneous wave
]

# New PDEs (none currently in our exp_5* set).
NEW_PDES = [
    "poisson_classic",        # 2D linear elliptic
    "heat_multiscale",        # 2D parabolic, multi-scale
    "heat_darcy",             # 2D parabolic, Darcy media (variable coef)
    "heat_complex",           # 2D parabolic, complex domain
    "heat_longtime",          # 2D parabolic, long-time
]

ALL_PDES = EXISTING_PDES + NEW_PDES
SEEDS = (42, 137, 256, 314, 529)


def run_one(pde_name: str, seed: int, results_root: Path,
            max_epochs: int = None):
    out_dir = results_root / pde_name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "phase_f_summary.json"

    if summary_path.exists():
        try:
            with open(summary_path) as f:
                rec = json.load(f)
            if rec.get("status") == "OK":
                print(f"  [SKIP] {pde_name} seed={seed} (already OK)")
                return rec
        except Exception:
            pass

    print("=" * 80)
    print(f"  Phase F — {pde_name} seed={seed}")
    print("=" * 80)

    from tier_5.exp_5_pinnacle_generic import run_experiment_pinnacle

    t0 = time.time()
    rec = {"pde_name": pde_name, "seed": seed}
    try:
        metrics, arch, cfg_dict, train_metrics = run_experiment_pinnacle(
            results_dir=str(out_dir),
            pde_name=pde_name,
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
        traceback.print_exc()
        rec.update({
            "status": f"ERROR: {type(e).__name__}: {e}",
            "elapsed_seconds": time.time() - t0,
            "max_epochs": max_epochs,
        })

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, default=str)
    print(f"  Saved: {summary_path}")
    print(f"  Elapsed: {rec['elapsed_seconds']:.1f}s\n")
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdes", nargs="+", default=None,
                        help="Subset of PDE names; default = all (existing + new)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Subset of seeds; default = all 5")
    parser.add_argument("--max-epochs", type=int, default=1500,
                        help="Per-run max epochs (default 1500 for safe convergence)")
    parser.add_argument("--results-dir", default="phase_f_results")
    args = parser.parse_args()

    pdes = args.pdes or ALL_PDES
    seeds = args.seeds or SEEDS

    results_root = (Path(_proj_root) / "experiments" / args.results_dir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    print(f"Phase F: {len(pdes)} PDEs x {len(seeds)} seeds = "
          f"{len(pdes) * len(seeds)} runs (max_epochs={args.max_epochs})")
    print(f"Output: {results_root}")

    n_ok = n_err = 0
    for pde in pdes:
        for seed in seeds:
            rec = run_one(pde, seed, results_root, max_epochs=args.max_epochs)
            if rec.get("status") == "OK":
                n_ok += 1
            else:
                n_err += 1

    print()
    print("=" * 80)
    print(f"Phase F summary: {n_ok} OK, {n_err} ERROR")
    print("=" * 80)


if __name__ == "__main__":
    main()
