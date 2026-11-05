#!/usr/bin/env python3
"""Compile the complete production ELPA wrapper with checked CP2K/ELPA mocks."""
from pathlib import Path
import os
import re
import shlex
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "src/fm/cp_fm_elpa.F"


def main():
    source = SOURCE.read_text()
    for path in (SOURCE, HERE / "mock_dependencies.F", HERE / "test_elpa.F"):
        lines = path.read_text().splitlines()
        assert all(len(line) <= 132 for line in lines), f"Overlong Fortran line in {path}"
        assert all(line == line.rstrip() for line in lines), f"Trailing whitespace in {path}"
        print(f"PASS {path.name}: maximum line length {max(map(len, lines))}", flush=True)

    constants = sorted(set(re.findall(r"\bELPA_2STAGE_REAL_\w+", source)))
    fc = shlex.split(os.environ.get("FC", "gfortran"))
    flags = ["-cpp", "-ffree-form", "-std=f2008", "-fopenmp", "-fcheck=all",
             "-Werror=line-truncation", "-g", "-O0"]
    with tempfile.TemporaryDirectory(prefix="elpa-tests-") as directory:
        build = Path(directory)
        (build / "fm").mkdir()
        (build / "base").mkdir()
        (build / "fm/cp_fm_elpa.F").write_text(source)
        (build / "base/base_uses.f90").write_text(
            'USE test_support, ONLY: verify, abort_test, warn_test, timeset, timestop\n'
            '#define CPASSERT(condition) CALL verify(condition, "CPASSERT")\n'
            '#define CPABORT(message) CALL abort_test(message)\n'
            '#define CPWARN_IF(condition, message) CALL warn_test(condition, message)\n'
            '#define MARK_USED(value)\n'
        )
        (build / "constants.f90").write_text(
            "MODULE elpa_constants\nIMPLICIT NONE\n"
            "INTEGER, PARAMETER :: ELPA_OK = 0, ELPA_SOLVER_1STAGE = 1, ELPA_SOLVER_2STAGE = 2\n"
            + "".join(f"INTEGER, PARAMETER :: {name} = {i}\n" for i, name in enumerate(constants, 10))
            + "END MODULE elpa_constants\n"
        )
        configurations = {
            "cpu": [],
            "hip": ["-D__OFFLOAD_HIP", "-D__HAS_IEEE_EXCEPTIONS"],
            "cuda": ["-D__OFFLOAD_CUDA"],
            "opencl": ["-D__OFFLOAD_OPENCL"],
        }
        for name, defines in configurations.items():
            exe = build / name
            subprocess.run(fc + flags + ["-D__ELPA", *defines, str(build / "constants.f90"),
                           str(HERE / "mock_dependencies.F"), str(build / "fm/cp_fm_elpa.F"),
                           str(HERE / "test_elpa.F"), "-o", str(exe)], cwd=build, check=True)
            for support_mask in range(8):
                subprocess.run([str(exe), str(support_mask)], cwd=build, check=True, timeout=30,
                               env=os.environ | {"OMP_NUM_THREADS": "1"})
            print(f"PASS {name}: all 8 GPU/OpenMP capability combinations", flush=True)
        subprocess.run(fc + flags + ["-c", str(build / "fm/cp_fm_elpa.F")], cwd=build, check=True)
        print("PASS compilation without ELPA", flush=True)


if __name__ == "__main__":
    main()
