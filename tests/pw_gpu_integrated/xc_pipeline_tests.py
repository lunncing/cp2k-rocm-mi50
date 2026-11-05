"""Compile the actual XC early dispatch, selector and C binding with small fixtures."""
from pathlib import Path
import os
import re
import shlex
import subprocess
from grid_ops_tests import routine

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run_xc_pipeline_tests(build, compiler, flags, include, blas, env, sanitize):
    wrapper = (ROOT / "src/pw/pw_gpu.F").read_text()
    bindings = "\n".join(routine(wrapper, name, True) for name in (
        "pw_gpu_fused_xc_available", "pw_gpu_xc_full_pipeline_available", "pw_gpu_xc_full_pipeline"))
    # The first two functions have an untyped FUNCTION declaration.
    (build / "xc_bindings.inc").write_text(bindings)
    xc = (ROOT / "src/xc/xc.F").read_text()
    caller = routine(xc, "xc_vxc_pw_create")
    early = caller.split("      bo = rho_r(1)%pw_grid%bounds_local", 1)[0]
    assert "pw_gpu_xc_full_pipeline(" in early
    early += "      cpu_calls = cpu_calls + 1\n      CALL timestop(handle)\n   END SUBROUTINE xc_vxc_pw_create\n"
    (build / "xc_dispatch.inc").write_text(early + routine(xc, "xc_vram_pbe_parameters", True))
    (build / "xc_cutoff.inc").write_text(routine(xc, "smooth_cutoff"))
    pbe = routine((ROOT / "src/xc/xc_pbe.F").read_text(), "pbe_lda_calc")
    pbe = pbe.replace("SUBROUTINE pbe_lda_calc", "SUBROUTINE cpu_pbe_lda_calc")
    pbe = re.sub(r"gpu_done = pbe_lda_eval_gpu\(.*?scale_ex\)", "gpu_done = .FALSE.", pbe, flags=re.S)
    (build / "xc_cpu_pbe.inc").write_text(pbe)
    fc = shlex.split(os.environ.get("FC", "gfortran"))
    fflags = ["-cpp", "-ffree-form", "-std=f2008", "-Werror=line-truncation", "-fcheck=all", "-fmax-errors=5",
              "-ffp-contract=off", "-fopenmp", "-I", str(build), "-J", str(build)]
    fflags += (["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
               if sanitize else ["-O2"])
    for name, defines in (("offload", ["-D__OFFLOAD"]), ("cpu", []),
                          ("no_pw_offload", ["-D__OFFLOAD", "-D__NO_OFFLOAD_PW"])):
        obj = build / (name + "_xc_bridge.o")
        subprocess.run(compiler + flags + include + defines + ["-c", str(HERE / "grid_test_bridge.c"),
                                                              "-o", str(obj)], check=True)
        objects = [str(obj)]
        if name == "offload":
            objects.append(str(build / "integral_mock.o"))
        exe = build / ("test_xc_pipeline_" + name)
        subprocess.run(fc + fflags + defines + [str(HERE / "test_xc_pipeline.F"),
                       *objects, *blas, "-o", str(exe)], check=True)
        for threads in (1, 3):
            subprocess.run([str(exe)], env=env | {"OMP_NUM_THREADS": str(threads)}, check=True, timeout=60)
        print(f"PASS compiled full XC dispatch/PBE/DFT comparisons ({name}, OMP=1/3)", flush=True)
