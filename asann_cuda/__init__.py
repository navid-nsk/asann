"""ASANN Custom CUDA Operations Package.

Provides drop-in CUDA replacements for all ASANN operations.
Build the extension first:
    cd asann_cuda && python setup.py install
"""

import os
import sys
import platform

# Ensure CUDA toolkit is on PATH/LD_LIBRARY_PATH for runtime libraries
if platform.system() == "Windows":
    _cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
    ]
    for _p in _cuda_paths:
        if os.path.isdir(_p) and _p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")
else:
    # Linux: CUDA is typically at /usr/local/cuda
    _cuda_paths = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/bin",
    ]
    for _p in _cuda_paths:
        if os.path.isdir(_p):
            _key = "LD_LIBRARY_PATH" if "lib" in _p else "PATH"
            if _p not in os.environ.get(_key, ""):
                os.environ[_key] = _p + os.pathsep + os.environ.get(_key, "")

# Also ensure PyTorch lib dir is on PATH (for torch DLLs / .so files)
try:
    import torch as _torch
    _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
    if os.path.isdir(_torch_lib) and _torch_lib not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
except ImportError:
    pass

try:
    import asann_cuda_ops
    CUDA_OPS_AVAILABLE = True
except ImportError:
    asann_cuda_ops = None
    CUDA_OPS_AVAILABLE = False

from .ops import *
