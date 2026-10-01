"""
Phase H: Tier 6 (graph node classification) multi-seed campaign.

Runs CiteSeer + PubMed for 5 seeds and aggregates results. Traffic
forecasting (METR-LA, PEMS-BAY) is left as single-seed since the user
asked specifically for the graph tier.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SEEDS = [42, 137, 256, 314, 529]
proj_root = Path(__file__).resolve().parent.parent.parent
tier6_dir = proj_root / "experiments" / "tier_6"
results_base = proj_root / "experiments" / "results" / "tier_6"
results_base.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {"key": "citeseer", "script": "exp_6a_citeseer.py", "label": "CiteSeer"},
    {"key": "pubmed",   "script": "exp_6b_pubmed.py",   "label": "PubMed"},
]


def find_results_dir(prefix: str):
    matches = sorted(results_base.glob(f"{prefix}*"))
    return matches[0] if matches else None


def run_one(dataset, seed):
    script_path = tier6_dir / dataset["script"]
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
            timeout=10800,  # 3h max
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
    prefix = f"{dataset['key']}_s{seed}"
    rdir = find_results_dir(prefix)
    if rdir is None:
        return None
    for fname in ("final_results.json", "results.json", "metrics.json"):
        f = rdir / fname
        if f.exists():
            try:
                return json.loads(f.read_text())
            except Exception:
                pass
    return None


campaign_log = results_base / "tier6_graph_multiseed_log.json"
per_run = []
for dataset in DATASETS:
    for seed in SEEDS:
        info = run_one(dataset, seed)
        info["dataset"] = dataset["key"]
        info["seed"] = seed
        per_run.append(info)
        campaign_log.write_text(json.dumps(per_run, indent=2))

print(f"\n{'=' * 70}\n  TIER 6 GRAPH MULTI-SEED SUMMARY\n{'=' * 70}")
agg = {"per_seed": {}, "summary": {}}
for dataset in DATASETS:
    agg["per_seed"][dataset["key"]] = {}
    metrics_per_seed = []
    for seed in SEEDS:
        m = collect_metrics(dataset, seed)
        if m:
            agg["per_seed"][dataset["key"]][str(seed)] = m
            metrics_per_seed.append(m)
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
        print(f"  {dataset['label']:<10} acc {mean:.4f} +/- {std:.4f}   CV={cv:.4f}   n={len(accs)}")
    else:
        print(f"  {dataset['label']:<10} no successful seeds")

agg_path = results_base / "tier6_graph_multiseed_summary.json"
agg_path.write_text(json.dumps(agg, indent=2))
print(f"\nSummary -> {agg_path}")
