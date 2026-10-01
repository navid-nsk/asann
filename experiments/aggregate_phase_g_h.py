"""
Standalone aggregator: scans tier_3 (image) and tier_6 (graph) per-seed
result directories on RunPod and prints a multi-seed CV table.

Run this anytime during/after the campaigns:
    python3 /workspace/experiments/aggregate_phase_g_h.py
"""
import json
import os
import statistics
from pathlib import Path

ROOT = Path(os.environ.get("RESULTS_ROOT",
                           "/workspace/experiments/results"))


def find_seed_dirs(tier_dir: Path):
    """Return {dataset_key: {seed: dir}}"""
    out = {}
    if not tier_dir.exists():
        return out
    for d in tier_dir.iterdir():
        if not d.is_dir():
            continue
        # Expect names like 'mnist_s42' or 'citeseer_s137'
        if "_s" not in d.name:
            continue
        key, seedstr = d.name.rsplit("_s", 1)
        try:
            seed = int(seedstr)
        except ValueError:
            continue
        out.setdefault(key, {})[seed] = d
    return out


def load_metrics(seed_dir: Path):
    f = seed_dir / "experiment_results.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def get_accuracy(payload):
    """Look in several common places for the test accuracy."""
    if not payload:
        return None
    m = payload.get("metrics", {}) or {}
    for k in ("test_accuracy", "accuracy", "test_acc"):
        v = m.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    # also try top-level
    for k in ("test_accuracy", "accuracy"):
        v = payload.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def get_arch_summary(payload):
    arch = (payload or {}).get("architecture", {}) or {}
    return {
        "num_layers": arch.get("num_layers"),
        "widths":     arch.get("widths"),
        "params":     arch.get("total_parameters"),
        "connections": arch.get("num_connections"),
    }


def cv(values):
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    if mean == 0:
        return None
    sd = statistics.stdev(values)
    return sd / abs(mean)


def report(tier_label, tier_dir):
    blocks = find_seed_dirs(tier_dir)
    print(f"\n{'=' * 78}")
    print(f"  {tier_label} ({tier_dir})")
    print(f"{'=' * 78}")
    if not blocks:
        print("  (no per-seed dirs found yet)")
        return None

    summary = {}
    for ds_key in sorted(blocks):
        seeds = blocks[ds_key]
        per_seed = {}
        for seed, d in sorted(seeds.items()):
            payload = load_metrics(d)
            acc = get_accuracy(payload)
            arch = get_arch_summary(payload)
            per_seed[seed] = {"accuracy": acc, "arch": arch,
                              "complete": payload is not None}
        accs = [v["accuracy"] for v in per_seed.values()
                if v["accuracy"] is not None]
        n_done = sum(1 for v in per_seed.values() if v["complete"])
        n_total = len(per_seed)
        if accs:
            mean = statistics.mean(accs)
            sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
            cv_pct = (sd / abs(mean) * 100) if mean else None
            line = f"  {ds_key:<22} acc {mean:.4f} +/- {sd:.4f}   CV={cv_pct:.2f}%   {n_done}/{n_total} seeds done"
        else:
            line = f"  {ds_key:<22} no completed seeds yet ({n_done}/{n_total})"
        print(line)
        # show per-seed
        for s, v in sorted(per_seed.items()):
            if v["complete"]:
                a = v["accuracy"]
                a_str = f"{a:.4f}" if a is not None else "n/a"
                arch = v["arch"]
                print(f"      seed={s:<4} acc={a_str}  layers={arch['num_layers']}  widths={arch['widths']}  params={arch['params']}  skips={arch['connections']}")
            else:
                print(f"      seed={s:<4} (in progress)")
        summary[ds_key] = {
            "n_done": n_done,
            "n_total": n_total,
            "accuracies": accs,
            "mean": statistics.mean(accs) if accs else None,
            "std": statistics.stdev(accs) if len(accs) > 1 else None,
            "cv": cv(accs) if len(accs) > 1 else None,
        }
    return summary


tier3 = report("TIER 3 (image classification)", ROOT / "tier_3")
tier6 = report("TIER 6 (graph node classification)", ROOT / "tier_6")

# Save aggregate
out = {"tier_3": tier3 or {}, "tier_6": tier6 or {}}
out_path = ROOT / "phase_g_h_aggregated.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nWrote aggregated summary -> {out_path}")
