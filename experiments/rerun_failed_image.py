"""
Rerun ONLY the seeds that failed in the main image multi-seed campaign:
  - KMNIST seeds 42, 137, 256 (download timeouts before we uploaded files)
  - SVHN  seeds 256, 314, 529 (surgery shape-mismatch bug, now patched)

Both groups should now succeed with the patched model.py and the
KMNIST data files in place.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

proj_root = Path(__file__).resolve().parent.parent
tier3_dir = proj_root / "experiments" / "tier_3"
results_base = proj_root / "experiments" / "results" / "tier_3"
results_base.mkdir(parents=True, exist_ok=True)

JOBS = [
    # KMNIST 5/5 already done. Only SVHN reruns remain (with the
    # trainer.py + model.py spatial-mismatch fix in place).
    ("svhn", "exp_3d_svhn.py", 256),
    ("svhn", "exp_3d_svhn.py", 314),
    ("svhn", "exp_3d_svhn.py", 529),
]


def find_results_json(prefix):
    matches = sorted(results_base.glob(f"{prefix}*"))
    for d in matches:
        f = d / "experiment_results.json"
        if f.exists():
            return f
    return None


campaign_log = results_base / "rerun_failed_image_log.json"
per_run = []
for ds, script_name, seed in JOBS:
    print(f"\n{'=' * 70}\n  {ds.upper()} seed {seed}\n{'=' * 70}")
    env = os.environ.copy()
    env["TIER_SEED"] = str(seed)
    env["RESULTS_SUFFIX"] = f"_s{seed}"
    start = time.time()
    try:
        r = subprocess.run(
            [sys.executable, str(tier3_dir / script_name)],
            env=env, cwd=str(proj_root), timeout=18000,
        )
        ok = r.returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    elapsed = time.time() - start

    info = {"dataset": ds, "seed": seed, "ok": ok, "elapsed_s": elapsed}
    per_run.append(info)
    campaign_log.write_text(json.dumps(per_run, indent=2))

# Final aggregate
print(f"\n{'=' * 70}\n  RERUN SUMMARY\n{'=' * 70}")
for info in per_run:
    flag = "OK" if info["ok"] else "FAILED"
    print(f"  {info['dataset']:<8} seed={info['seed']:<4} {flag} ({info['elapsed_s']/60:.1f} min)")
