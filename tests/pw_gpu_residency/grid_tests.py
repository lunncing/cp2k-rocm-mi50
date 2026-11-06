"""Production Grid buffer lifetime/bridge and Fortran caller tests.

The Grid/offload modules and qs_integrate_potential_product.F were absent from
this snapshot. They were restored from CP2K commit
37370f362787e9d04d7644023de78707537d81e5, the same revision as the other restored
callers, before adding the pass-through changes. Only backend arithmetic and
unrelated Fortran dependencies are mocked here; the changed C/C++ dispatch,
buffer ownership, Fortran bindings and transfer decisions are compiled.
These CPU tests do not measure MI50 step timings or replace a full HIP build.
"""
import os
import shlex
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run_grid_tests(build, cc, flags, include, blas, env):
    off = build / "grid_offload"
    off.mkdir()
    for name in ("offload_buffer.c", "offload_buffer.h"):
        shutil.copy2(ROOT / "src/offload" / name, off / name)
    # Map the runtime's default stream to a checked mock stream. Grid/PW
    # producers must still synchronize explicitly before crossing streams.
    (off / "offload_runtime.h").write_text('''
#include "mock_offload.h"
#define offloadMemcpyAsyncDtoH(d,s,n,st) offloadMemcpyAsyncDtoH(d,s,n,(st)?(st):1)
#define offloadMemcpyAsyncHtoD(d,s,n,st) offloadMemcpyAsyncHtoD(d,s,n,(st)?(st):1)
#define offloadStreamSynchronize(st) offloadStreamSynchronize((st)?(st):1)
''')
    (off / "offload_library.h").write_text('#include "mock_offload.h"\n')
    (off / "offload_mempool.h").write_text('''
#include "mock_offload.h"
#include <stdlib.h>
static inline void *offload_mempool_host_malloc(size_t n) { return malloc(n); }
static inline void offload_mempool_host_free(void *p) { free(p); }
static inline void *offload_mempool_device_malloc(size_t n) {
#ifdef __OFFLOAD
  void *p; offloadMalloc(&p, n); return p;
#else
  (void)n; return NULL;
#endif
}
static inline void offload_mempool_device_free(void *p) {
#ifdef __OFFLOAD
  offloadFree(p);
#else
  (void)p;
#endif
}
''')
    inc = include + ["-I", str(off), "-I", str(build)]
    obj = build / "grid_buffer.o"
    subprocess.run(cc + flags + inc + ["-D__OFFLOAD", "-c", str(off / "offload_buffer.c"),
                                      "-o", str(obj)], check=True)
    exe = build / "test_grid_passthrough"
    subprocess.run(cc + flags + inc + ["-D__OFFLOAD", str(HERE / "test_grid_passthrough.c"), str(obj),
                   str(HERE.parent / "pw_gpu_integrated/mock_offload.c"), *blas, "-lm", "-o", str(exe)], check=True)
    for threads in ("1", "3"):
        subprocess.run([str(exe)], env=env | {"OMP_NUM_THREADS": threads}, check=True, timeout=60)

    from grid_ops_tests import routine
    wrapper = (ROOT / "src/pw/pw_gpu.F").read_text()
    functions = ("res_grid_ok", "res_rs_ok", "res_gs_ok", "pw_gpu_res_enabled", "pw_gpu_res_density",
                 "pw_gpu_res_density_grid", "pw_gpu_res_potential")
    subs = ("pw_gpu_res_ensure_host_ptr", "pw_gpu_res_ensure_host_r", "pw_gpu_res_ensure_host_g")
    (build / "grid_pw_bindings.inc").write_text(
        "\n".join(routine(wrapper, n, True) for n in functions) +
        "\n".join(routine(wrapper, n) for n in subs))
    wrapper = (ROOT / "src/pw_env/rs_pw_interface.F").read_text()
    (build / "grid_rs_bindings.inc").write_text(routine(wrapper, "grid_layout_ok", True) +
        "\n".join(routine(wrapper, n) for n in ("rs_pw_grid_prepare", "density_rs2pw", "potential_pw2rs")))
    wrapper = (ROOT / "src/offload/offload_api.F").read_text()
    (build / "grid_offload_bindings.inc").write_text(
        "\n".join(routine(wrapper, n, True) for n in ("offload_grid_device", "offload_grid_bind",
                                                     "offload_get_buffer_device_pointer")) +
        "\n".join(routine(wrapper, n) for n in ("offload_create_buffer", "offload_free_buffer",
                                               "offload_grid_prepare", "offload_grid_host")))
    fc = shlex.split(os.environ.get("FC", "gfortran"))
    fflags = ["-cpp", "-std=f2008", "-Werror=line-truncation", "-fcheck=all", "-fopenmp",
              "-ffp-contract=off", "-O1", "-g", "-fsanitize=address,undefined",
              "-fno-omit-frame-pointer", "-J", str(build), "-I", str(build)]
    for mode, defines in (("offload", ["-D__OFFLOAD"]), ("cpu", []),
                           ("no_pw", ["-D__OFFLOAD", "-D__NO_OFFLOAD_PW"])):
        objects = []
        for name, src in (("buffer", off / "offload_buffer.c"),
                          ("bridge", HERE / "grid_fortran_bridge.c"),
                          ("mock", HERE.parent / "pw_gpu_integrated/mock_offload.c")):
            obj = build / (mode + "_grid_" + name + ".o")
            subprocess.run(cc + flags + inc + defines + ["-c", str(src), "-o", str(obj)], check=True)
            objects.append(str(obj))
        exe = build / ("test_grid_bindings_" + mode)
        subprocess.run(fc + fflags + defines + [str(HERE / "test_grid_bindings.F90"), *objects, *blas,
                                               "-o", str(exe)], check=True)
        for threads in ("1", "3"):
            for enabled in ("0", "1"):
                subprocess.run([str(exe)], env=env | {"OMP_NUM_THREADS": threads, "CP2K_PW_GPU_RES": enabled},
                               check=True, timeout=60)
        print(f"PASS Grid/PW Fortran bindings/callers ({mode}, RES=0/1, OMP=1/3)")

    source = (ROOT / "src/grid/gpu/grid_gpu_context.cu").read_text()
    calls = []
    for name in ("grid_gpu_collocate_task_list", "grid_gpu_integrate_task_list"):
        first = source.index('extern "C" void ' + name + '(')
        brace = source.index('{', first)
        depth, last = 1, brace + 1
        while depth:
            depth += (source[last] == '{') - (source[last] == '}')
            last += 1
        calls.append(source[first:last])
    (build / "grid_gpu_calls.inc").write_text("\n".join(calls))
    cxx = shlex.split(os.environ.get("CXX", "g++"))
    cxxflags = [f for f in flags if f != "-std=c11"] + ["-std=c++17"]
    exe = build / "test_grid_gpu_driver"
    subprocess.run(cxx + cxxflags + inc + [str(HERE / "test_grid_gpu_driver.cpp"),
        str(build / "offload_grid_buffer.o"), str(build / "offload_grid_mock.o"), *blas,
        "-o", str(exe)], check=True)
    subprocess.run([str(exe)], env=env, check=True, timeout=60)

    source = (ROOT / "src/grid/grid_task_list.c").read_text()
    calls = []
    for name in ("grid_collocate_task_list", "grid_integrate_task_list"):
        first = source.index('void ' + name + '(')
        brace = source.index('{', first)
        depth, last = 1, brace + 1
        while depth:
            depth += (source[last] == '{') - (source[last] == '}')
            last += 1
        calls.append(source[first:last])
    (build / "grid_dispatch_calls.inc").write_text("\n".join(calls))
    exe = build / "test_grid_dispatch"
    subprocess.run(cc + flags + inc + ["-D__OFFLOAD_HIP", str(HERE / "test_grid_dispatch.c"),
        str(build / "offload_grid_buffer.o"), str(build / "offload_grid_mock.o"), *blas, "-lm",
        "-o", str(exe)], check=True)
    subprocess.run([str(exe)], env=env, check=True, timeout=60)
