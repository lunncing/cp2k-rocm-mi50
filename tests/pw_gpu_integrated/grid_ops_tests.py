"""Compile actual grid/Hartree/PBE callers and C bindings with dependency stubs."""
from pathlib import Path
import os
import re
import shlex
import subprocess
from generate_pbe_kernel import generate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def routine(source, name, function=False):
    word = "FUNCTION" if function else "SUBROUTINE"
    match = re.search(r"^ *(?:LOGICAL )?" + word + " " + re.escape(name) +
                      r"\(.*?^ *END " + word + " " + re.escape(name) + r"[ \t]*$",
                      source, re.M | re.S)
    assert match, name
    return match[0] + "\n"


def run_grid_ops_tests(build, compiler, flags, include, blas, env, sanitize):
    import fypp
    assert (ROOT / "src/pw/gpu/pw_gpu_pbe.h").read_text() == generate()
    wrapper = (ROOT / "src/pw/pw_gpu.F").read_text()
    names = ["pw_gpu_copy", "pw_gpu_axpy", "pw_gpu_remap_copy", "pw_gpu_remap_axpy",
             "modify_hartree_pot_gpu", "pbe_lda_eval_gpu"]
    bindings = "\n".join(routine(wrapper, name, True) for name in names)
    (build / "grid_bindings.inc").write_text(bindings)
    methods = (ROOT / "src/pw/pw_methods.F").read_text()
    for name in ("copy", "axpy"):
        # There is a space before '(' in the Fypp template.
        templ = routine(methods.replace("$_${space}$ (", "$_${space}$("),
                        "pw_" + name + "_${kind}$_${kind2}$_${space}$")
        # Host-resident copy/AXPY must stay on the CPU, even in offload builds.
        assert not re.search(r"pw_gpu_(?:remap_)?" + name + r"\b", templ)
        ref = templ.replace("pw_" + name + "_${kind}", "cpu_" + name + "_${kind}")
        ref = ref.replace("#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)", "#if 0")
        (build / (name + ".inc")).write_text(templ + ref)
    for file, name in (("src/xc/xc_pbe.F", "pbe_lda_calc"),
                       ("src/cp_ddapc_util.F", "modify_hartree_pot")):
        templ = routine((ROOT / file).read_text(), name)
        ref = templ.replace("SUBROUTINE " + name, "SUBROUTINE cpu_" + name)
        ref = ref.replace("#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)", "#if 0")
        if name == "pbe_lda_calc":
            ref = re.sub(r"gpu_done = pbe_lda_eval_gpu\(.*?scale_ex\)", "gpu_done = .FALSE.", ref, flags=re.S)
        (build / (name + ".inc")).write_text(templ + ref)
    options = fypp.FyppOptions()
    options.includes = [str(build)]
    options.line_length = 132
    generated = build / "test_grid_ops.F90"
    fypp.Fypp(options).process_file(str(HERE / "test_grid_ops.F"), str(generated))
    assert max(map(len, generated.read_text().splitlines())) <= 132
    fc = shlex.split(os.environ.get("FC", "gfortran"))
    fflags = ["-cpp", "-std=f2008", "-Werror=line-truncation", "-fcheck=all", "-fmax-errors=5",
              "-ffp-contract=off", "-fopenmp", "-J", str(build)]
    fflags += (["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
               if sanitize else ["-O2"])
    for name, defines in (("offload", ["-D__OFFLOAD"]), ("cpu", []),
                          ("no_pw_offload", ["-D__OFFLOAD", "-D__NO_OFFLOAD_PW"])):
        obj = build / (name + "_grid.o")
        subprocess.run(compiler + flags + include + defines + ["-c", str(HERE / "grid_test_bridge.c"),
                                                              "-o", str(obj)], check=True)
        objects = [str(obj)]
        if name == "offload":
            objects.append(str(build / "integral_mock.o"))
        exe = build / ("test_grid_" + name)
        subprocess.run(fc + fflags + defines + [str(generated), *objects, *blas, "-o", str(exe)], check=True)
        for threads in (1, 3):
            subprocess.run([str(exe)], env=env | {"OMP_NUM_THREADS": str(threads)},
                           check=True, timeout=60)
        print(f"PASS compiled grid/Hartree/PBE CPU comparisons ({name}, OMP=1/3)", flush=True)
