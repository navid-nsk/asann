import os
import platform
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the directory containing this setup.py
this_dir = os.path.dirname(os.path.abspath(__file__))
kernels_dir = os.path.join(this_dir, "kernels")

# Collect all .cu source files
cuda_sources = [
    os.path.join(kernels_dir, f)
    for f in sorted(os.listdir(kernels_dir))
    if f.endswith(".cu")
]

# Main binding file
binding_sources = [
    os.path.join(this_dir, "bindings", "asann_cuda_ops.cpp"),
]

all_sources = binding_sources + cuda_sources

# Platform-specific C++ compiler flags
if platform.system() == "Windows":
    cxx_flags = ["/O2"]
else:
    cxx_flags = ["-O2"]

setup(
    name="asann_cuda_ops",
    ext_modules=[
        CUDAExtension(
            name="asann_cuda_ops",
            sources=all_sources,
            include_dirs=[kernels_dir],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": [
                    "-O2",
                    "--use_fast_math",
                    "-gencode=arch=compute_60,code=sm_60",
                    "-gencode=arch=compute_70,code=sm_70",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_90,code=sm_90",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
