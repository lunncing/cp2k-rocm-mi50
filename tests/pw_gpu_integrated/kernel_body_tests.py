"""Compile the .cu bodies as C++ and emulate launch geometry without a GPU.

Only CUDA launch syntax and __global__/__device__ annotations are substituted.
This checks the production indexing/launchers, complementing the checked runtime
mock and the independent compiled Fortran numerical comparisons. It is not a HIP
compiler or hardware test.
"""
from pathlib import Path
import os
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run_kernel_body_tests(build, flags, include, env):
    source = (ROOT / "src/pw/gpu/pw_gpu_kernels.cu").read_text()
    source, count = re.subn(r"(\w+)<<<(.*?)>>>\(", r"emulate(\1, \2, ", source, flags=re.S)
    assert count == 12, count
    prelude = '''#include "mock_offload.h"
#define __global__
#define __device__
struct dim { unsigned int x; };
static dim blockIdx, threadIdx, blockDim, gridDim;
template <class Kernel, class... Args>
void emulate(Kernel kernel, unsigned int blocks, unsigned int threads,
             int shared, offloadStream_t stream, Args... args) {
  (void)shared; (void)stream;
  blockDim.x = threads; gridDim.x = blocks;
  for (blockIdx.x = 0; blockIdx.x < blocks; blockIdx.x++)
    for (threadIdx.x = 0; threadIdx.x < threads; threadIdx.x++) kernel(args...);
}
'''
    path = build / "src/pw/gpu/emulated_kernels.cpp"
    path.write_text(prelude + source + (HERE / "test_kernel_bodies.cpp").read_text())
    cxx = shlex.split(os.environ.get("CXX", "c++"))
    cxxflags = [f for f in flags if f != "-std=c11"] + ["-std=c++17"]
    exe = build / "test_kernel_bodies"
    subprocess.run(cxx + cxxflags + include + ["-D__OFFLOAD", str(path), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], env=env, check=True, timeout=60)
