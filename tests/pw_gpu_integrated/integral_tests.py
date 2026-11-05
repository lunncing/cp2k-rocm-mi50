"""Compile the production integral template with minimal CP2K dependency stubs."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run_integral_tests(build, compiler, flags, include, blas, env, sanitize):
    import fypp

    source = (ROOT / "src/pw/pw_methods.F").read_text()
    template = re.search(
        r"^ *FUNCTION pw_integral_ab_\$\{kind\}.*?"
        r"^ *END FUNCTION pw_integral_ab_\$\{kind\}.*?$",
        source, re.MULTILINE | re.DOTALL,
    )
    assert template is not None
    (build / "pw_integral_ab.inc").write_text(template[0] + "\n")
    options = fypp.FyppOptions()
    options.includes = [str(build)]
    options.line_length = 132
    generated = build / "test_pw_integral.F90"
    fypp.Fypp(options).process_file(str(HERE / "test_pw_integral.F"), str(generated))
    assert max(map(len, generated.read_text().splitlines())) <= 132
    fc = shlex.split(os.environ.get("FC", "gfortran"))
    fflags = ["-cpp", "-std=f2008", "-Wall", "-Wextra", "-Werror=line-truncation",
              "-fcheck=all", "-ffp-contract=off", "-J", str(build)]
    fflags += (["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
               if sanitize else ["-O2"])
    for name, defines in (("offload", ["-D__OFFLOAD"]), ("cpu", []),
                          ("no_pw_offload", ["-D__OFFLOAD", "-D__NO_OFFLOAD_PW"])):
        obj = build / f"pw_integral_{name}.o"
        subprocess.run(compiler + flags + include + defines + [
            "-c", str(build / "src/pw/gpu/pw_gpu_internal.c"), "-o", str(obj)], check=True)
        objects = [str(obj)]
        if name == "offload":
            mock = build / "integral_mock.o"
            subprocess.run(compiler + flags + include + ["-c", str(HERE / "mock_offload.c"),
                                                       "-o", str(mock)], check=True)
            objects.append(str(mock))
        exe = build / f"test_pw_cpu_dot_{name}"
        subprocess.run(compiler + flags + [str(HERE / "test_pw_cpu_dot.c"), *objects,
                                          *blas, "-lm", "-o", str(exe)], check=True)
        subprocess.run([str(exe)], env=env, check=True, timeout=60)
        exe = build / f"test_pw_integral_{name}"
        subprocess.run(fc + fflags + defines + [str(generated), *objects, *blas,
                                                "-fopenmp", "-o", str(exe)], check=True)
        subprocess.run([str(exe)], env=env, check=True, timeout=60)
        result = subprocess.run([str(exe), "incompatible"], env=env, text=True,
                                capture_output=True, timeout=60)
        assert result.returncode != 0 and "Grids incompatible" in result.stderr, result
        assert "AddressSanitizer" not in result.stderr and "runtime error:" not in result.stderr, result.stderr
        print(f"PASS C DDOT and all 16 Fortran integral variants ({name})", flush=True)
