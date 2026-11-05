"""Compile the production residency bindings and host access paths with fixtures."""
import os
import re
import shlex
import subprocess
from pathlib import Path

from grid_ops_tests import routine

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run_fortran_tests(build, cc, flags, include, blas, env):
    import fypp

    wrapper = (ROOT / "src/pw/pw_gpu.F").read_text()
    functions = ("pw_gpu_res_enabled", "pw_gpu_res_integrate_r", "res_grid_ok", "res_rs_ok",
                 "pw_gpu_res_vtot_prepare", "pw_gpu_res_vtot_skip_host")
    subroutines = ("pw_gpu_res_ensure_host_ptr", "pw_gpu_res_ensure_host_r", "pw_gpu_res_ensure_host_g",
                   "pw_gpu_res_set_lazy", "pw_gpu_res_defer_density", "pw_gpu_res_discard")
    bindings = "\n".join(routine(wrapper, n, True) for n in functions)
    bindings += "\n".join(routine(wrapper, n) for n in subroutines)
    (build / "res_bindings.inc").write_text(bindings)
    methods = (ROOT / "src/pw/pw_methods.F").read_text().replace("$ (", "$(")
    for name, template, function in (
        ("host_integrate", "pw_integrate_function_${kind}$_${space}$", True),
        ("host_copy", "pw_copy_${kind}$_${kind2}$_${space}$", False),
        ("host_axpy", "pw_axpy_${kind}$_${kind2}$_${space}$", False),
        ("host_zero", "pw_zero_${kind}$_${space}$", False),
    ):
        (build / (name + ".inc")).write_text(routine(methods, template, function))
    caller = routine((ROOT / "src/qs_collocate_density.F").read_text(), "calculate_rho_elec")
    charge = re.search(r"^      IF \(PRESENT\(total_rho\)\) THEN.*?^      END IF", caller, re.S | re.M)
    assert charge is not None
    (build / "charge.inc").write_text(charge[0] + "\n")
    options = fypp.FyppOptions()
    options.includes = [str(build)]
    options.line_length = 132
    generated = build / "test_res_bindings.F90"
    fypp.Fypp(options).process_file(str(HERE / "test_res_bindings.F"), str(generated))
    fc = shlex.split(os.environ.get("FC", "gfortran"))
    fflags = ["-cpp", "-std=f2008", "-Werror=line-truncation", "-fcheck=all", "-fopenmp",
              "-ffp-contract=off", "-O1", "-g", "-fsanitize=address,undefined",
              "-fno-omit-frame-pointer", "-J", str(build)]
    for name, defines in (("offload", ["-D__OFFLOAD"]), ("cpu", []),
                          ("no_pw_offload", ["-D__OFFLOAD", "-D__NO_OFFLOAD_PW"])):
        obj = build / (name + "_res_bridge.o")
        subprocess.run(cc + flags + include + defines + ["-c", str(HERE / "fortran_bridge.c"),
                                                        "-o", str(obj)], check=True)
        objects = [str(obj)]
        if name == "offload":
            mock = build / "res_mock.o"
            subprocess.run(cc + flags + include + ["-c", str(HERE.parent / "pw_gpu_integrated/mock_offload.c"),
                                                   "-o", str(mock)], check=True)
            objects.append(str(mock))
        exe = build / ("test_res_bindings_" + name)
        subprocess.run(fc + fflags + defines + [str(generated), *objects, *blas, "-o", str(exe)], check=True)
        for threads in ("1", "3"):
            for enabled in ("0", "1"):
                subprocess.run([str(exe)], env=env | {"OMP_NUM_THREADS": threads, "CP2K_PW_GPU_RES": enabled},
                               check=True, timeout=60)
        print(f"PASS residency Fortran bindings/charge/host consumers ({name}, RES=0/1, OMP=1/3)")
