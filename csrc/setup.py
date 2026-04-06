"""Build MineInsight CUDA extensions.

Usage:
    cd /mnt/forge-data/modules/05_wave9/21_MineInsight
    source .venv/bin/activate
    python csrc/setup.py build_ext --inplace

Target: L4 (sm_89), CUDA 12.8, PyTorch cu128
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Force sm_89 for L4 GPUs
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

setup(
    name="mineinsight_cuda_ops",
    version="1.0.0",
    ext_modules=[
        CUDAExtension(
            name="mineinsight_cuda_ops",
            sources=["csrc/mineinsight_cuda_ops.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_89,code=sm_89",
                    "--ptxas-options=-v",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
