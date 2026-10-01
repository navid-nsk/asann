"""
One-shot patcher to be run on RunPod /workspace/ after uploading the
experiment files. Three patches:

1. Add CSANN* -> ASANN* class aliases to /workspace/asann/__init__.py
   so the existing experiment scripts that import 'from csann import
   CSANNConfig, CSANNModel, CSANNTrainer, CSANNOptimizerConfig' resolve
   under the new asann package.

2. Modify experiments/common.py wrapper functions to honor a
   RESULTS_SUFFIX env var, so each multi-seed run gets its own
   per-seed results directory.

3. Insert a seed-setup block at the top of every tier_3 exp_3*.py and
   tier_6 exp_6a_citeseer.py / exp_6b_pubmed.py, reading TIER_SEED from
   the environment.

Run with: python setup_runpod_patches.py /workspace
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace")


# ---------------------------------------------------------------------------
# 1. CSANN class aliases
# ---------------------------------------------------------------------------
asann_init = ROOT / "asann" / "__init__.py"
init_text = asann_init.read_text()

ALIAS_BLOCK = """
# ============================================================================
# Backward-compat class aliases: experiment scripts use the old CSANN* names
# ============================================================================
CSANNConfig = ASANNConfig
CSANNOptimizerConfig = ASANNOptimizerConfig
CSANNModel = ASANNModel
CSANNTrainer = ASANNTrainer
CSANNLoss = ASANNLoss
CSANNLRController = ASANNLRController
CSANNWarmupScheduler = ASANNWarmupScheduler
CSANNOptimizer = ASANNOptimizer
"""

if "CSANNConfig = ASANNConfig" not in init_text:
    init_text = init_text.rstrip() + "\n" + ALIAS_BLOCK + "\n"
    asann_init.write_text(init_text)
    print(f"[1] Added CSANN* aliases to {asann_init}")
else:
    print(f"[1] CSANN* aliases already present in {asann_init}")


# ---------------------------------------------------------------------------
# 2. Patch experiments/common.py to honor RESULTS_SUFFIX env var
# ---------------------------------------------------------------------------
common_py = ROOT / "experiments" / "common.py"
common_text = common_py.read_text()

# Pattern: results_dir = results_base_dir / name.lower().replace(" ", "_").replace("/", "_")
# Becomes: results_dir = results_base_dir / (name.lower().replace(" ", "_").replace("/", "_") + os.environ.get("RESULTS_SUFFIX", ""))
old_pat = 'results_dir = results_base_dir / name.lower().replace(" ", "_").replace("/", "_")'
new_pat = 'results_dir = results_base_dir / (name.lower().replace(" ", "_").replace("/", "_") + os.environ.get("RESULTS_SUFFIX", ""))'
n = common_text.count(old_pat)
if n:
    common_text = common_text.replace(old_pat, new_pat)
    common_py.write_text(common_text)
    print(f"[2] Patched {n} wrapper(s) in {common_py} to honor RESULTS_SUFFIX")
else:
    print(f"[2] common.py already patched OR pattern not found")


# ---------------------------------------------------------------------------
# 3. Add seed-setup block to each exp_3*.py and exp_6a/b.py
# ---------------------------------------------------------------------------
SEED_BLOCK = """# === Multi-seed campaign: read TIER_SEED from env, default 42 ===
import os as _os_seed
import random as _random_seed
_T_SEED = int(_os_seed.environ.get("TIER_SEED", "42"))
_random_seed.seed(_T_SEED)
try:
    import numpy as _np_seed
    _np_seed.random.seed(_T_SEED)
except ImportError:
    pass
try:
    import torch as _torch_seed
    _torch_seed.manual_seed(_T_SEED)
    if _torch_seed.cuda.is_available():
        _torch_seed.cuda.manual_seed_all(_T_SEED)
except ImportError:
    pass
# === end seed setup ===
"""

target_files = []
for pat in ("exp_3*.py",):
    target_files.extend((ROOT / "experiments" / "tier_3").glob(pat))
for fname in ("exp_6a_citeseer.py", "exp_6b_pubmed.py"):
    p = ROOT / "experiments" / "tier_6" / fname
    if p.exists():
        target_files.append(p)

for f in target_files:
    txt = f.read_text()
    if "_T_SEED" in txt:
        print(f"[3] {f.name}: already patched")
        continue
    # Insert SEED_BLOCK after the docstring (first triple-quoted block) or at top.
    # Heuristic: split off the leading triple-quoted docstring and put block right after.
    m = re.match(r'(\s*"""[\s\S]*?"""\s*\n)', txt)
    if m:
        head = m.group(1)
        rest = txt[len(head):]
        new_txt = head + SEED_BLOCK + rest
    else:
        new_txt = SEED_BLOCK + txt
    f.write_text(new_txt)
    print(f"[3] {f.name}: inserted seed setup block")

print("\nAll patches applied.")
