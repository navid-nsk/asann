"""
Phase G: Tier 3 (image classification) multi-seed campaign.

Runs each exp_3*.py for 5 seeds and aggregates results.

Each experiment is launched as a subprocess with TIER_SEED set in the
environment. The exp scripts read TIER_SEED at the top, fall back to 42
if not set, and stamp it into the data-split seed and torch.manual_seed.

Results land in experiments/results/tier_3/<dataset>_s<seed>/.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SEEDS = [42, 137, 256, 314, 529]
proj_root = Path(__file__).resolve().parent.parent.parent
tier3_dir = proj_root / "experiments" / "tier_3"
results_base = proj_root / "experiments" / "results" / "tier_3"
results_base.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {"key": "mnist",         "script": "exp_3a_mnist.py",         "label": "MNIST"},
    {"key": "kmnist",        "script": "exp_3c_kmnist.py",        "label": "KMNIST"},
    {"key": "fashion_mnist", "script": "exp_3b_fashion_mnist.py", "label": "Fashion-MNIST"},
    {"key": "svhn",          "script": "exp_3d_svhn.py",          "label": "SVHN"},
    {"key": "cifar10",       "script": "exp_3e_cifar10.py",       "label": "CIFAR-10"},
    {"key": "cifar100",      "script": "exp_3f_cifar100.py",      "label": "CIFAR-100"},
    # STL-10 omitted from multi-seed (very long training); single-seed result retained.
]


def find_results_dir(prefix: str) -> Path | None:
    """Find the single results dir matching prefix (e.g., 'mnist_s42')."""
    matches = sorted(results_base.glob(f"{prefix}*"))
    return matches[0] if matches else None


def run_one(dataset, seed):
    script_path = tier3_dir / dataset["script"]
    label = dataset["label"]
    print(f"\n{'=' * 70}")
    print(f"  {label}  seed {seed}")
    print(f"{'=' * 70}")

    env = os.environ.copy()
    env["TIER_SEED"] = str(seed)
    env["RESULTS_SUFFIX"] = f"_s{seed}"

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            env=env,
            cwd=str(proj_root),
            timeout=18000,  # 5 hours max per run
        )
        elapsed = time.time() - start
        ok = result.returncode == 0
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        ok = False
        print(f"  TIMEOUT after {elapsed/60:.1f} min")
    except Exception as e:
        elapsed = time.time() - start
        ok = False
        print(f"  EXCEPTION: {e}")

    return {"ok": ok, "elapsed_s": elapsed}


def collect_metrics(dataset, seed):
    """Look for the per-seed results dir and pull final metrics."""
    prefix = f"{dataset['key']}_s{seed}"
    rdir = find_results_dir(prefix)
    if rdir is None:
        return None
    # Standard wrapper writes 'final_results.json' or similar
    for fname in ("final_results.json", "results.json", "metrics.json"):
        f = rdir / fname
        if f.exists():
            try:
                return json.loads(f.read_text())
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
campaign_log = results_base / "tier3_multiseed_log.json"
campaign_log.parent.mkdir(parents=True, exist_ok=True)

per_run = []
for dataset in DATASETS:
    for seed in SEEDS:
        info = run_one(dataset, seed)
        info["dataset"] = dataset["key"]
        info["seed"] = seed
        per_run.append(info)
        # checkpoint progress every run
        campaign_log.write_text(json.dumps(per_run, indent=2))

# Aggregate
print(f"\n{'=' * 70}")
print(f"  TIER 3 MULTI-SEED SUMMARY")
print(f"{'=' * 70}")

agg = {"per_seed": {}, "summary": {}}
for dataset in DATASETS:
    agg["per_seed"][dataset["key"]] = {}
    metrics_per_seed = []
    for seed in SEEDS:
        m = collect_metrics(dataset, seed)
        if m:
            agg["per_seed"][dataset["key"]][str(seed)] = m
            metrics_per_seed.append(m)
    if metrics_per_seed:
        # extract accuracy values
        accs = []
        for m in metrics_per_seed:
            v = m.get("accuracy") or m.get("test_accuracy") or m.get("metrics", {}).get("accuracy")
            if isinstance(v, (int, float)):
                accs.append(float(v))
        if accs:
            mean = float(np.mean(accs))
            std = float(np.std(accs))
            cv = float(std / abs(mean)) if mean else None
            agg["summary"][dataset["key"]] = {
                "n_ok": len(accs),
                "accuracy_mean": mean,
                "accuracy_std": std,
                "accuracy_cv": cv,
            }
            print(f"  {dataset['label']:<15} acc {mean:.4f} +/- {std:.4f}   CV={cv:.4f}   n={len(accs)}")
        else:
            print(f"  {dataset['label']:<15} no accuracy field found")
    else:
        print(f"  {dataset['label']:<15} 0 successful seeds")

agg_path = results_base / "tier3_multiseed_summary.json"
agg_path.write_text(json.dumps(agg, indent=2))
print(f"\nSummary -> {agg_path}")
