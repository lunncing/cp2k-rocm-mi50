"""Compile the production single-rank Cholesky paths with checked HIP/BLAS mocks."""
import os
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run_cholesky_tests(build, compiler, flags, include, blas, environment, sanitize):
    environment = {name: value for name, value in environment.items()
                   if name != "CP2K_CHOLESKY_FAST_SINGLE_RANK"}
    environment |= {"CP2K_PW_GPU_MEM_SAMPLE": "0", "CP2K_PW_GPU_RES_STATS": "0", "OMP_DYNAMIC": "FALSE"}
    variants = {
        "hip": ["-D__OFFLOAD_HIP"],
        "no-rocblas": ["-D__OFFLOAD_HIP", "-DCHOLESKY_NO_ROCBLAS"],
        "no-blas-get": ["-D__OFFLOAD_HIP", "-DCHOLESKY_NO_BLAS_GET"],
        "no-blas-set": ["-D__OFFLOAD_HIP", "-DCHOLESKY_NO_BLAS_SET"],
        "non-hip": [],
    }
    for name, defines in variants.items():
        executable = build / f"cholesky-{name}"
        subprocess.run(compiler + flags + include + ["-D__OFFLOAD", *defines,
                       "-include", str(HERE / "mock_cholesky.h"), str(HERE / "test_cholesky.c"),
                       str(HERE / "mock_cholesky.c"), str(HERE / "mock_offload.c"),
                       "-Wl,--export-dynamic", *blas, "-ldl", "-lm", "-o", str(executable)], check=True)
        for setting in ("default", "0", "1", "FALSE", "OFF"):
            overrides = {} if setting == "default" else {"CP2K_CHOLESKY_FAST_SINGLE_RANK": setting}
            enabled = int(name not in ("no-rocblas", "non-hip") and setting in ("default", "1"))
            result = subprocess.run([str(executable), str(enabled)], env=environment | overrides,
                                    text=True, capture_output=True, timeout=120)
            assert result.returncode == 0, (name, setting, result.stdout, result.stderr)
            assert "AddressSanitizer" not in result.stderr and "runtime error:" not in result.stderr, result.stderr
            print(f"PASS Cholesky {name}, gate={setting}: C numerics, strides, rollback and threads", flush=True)
    run_fortran_cholesky_tests(build, compiler, flags, include, blas, environment, sanitize)


def run_fortran_cholesky_tests(build, compiler, flags, include, blas, environment, sanitize):
    directory = build / "cholesky-fortran"
    (directory / "fm").mkdir(parents=True)
    (directory / "base").mkdir()
    source = ROOT / "src/fm/cp_fm_cholesky.F"
    for path in (source, HERE / "test_cholesky.F", HERE / "mock_cholesky_dependencies.F"):
        assert all(len(line) <= 132 for line in path.read_text().splitlines()), path
    (directory / "fm/cp_fm_cholesky.F").write_text(source.read_text())
    (directory / "base/base_uses.f90").write_text(
        'USE cholesky_test_support, ONLY: verify, abort_test, cp_abort, timeset, timestop\n'
        '#define CPASSERT(condition) CALL verify(condition, "CPASSERT")\n'
        '#define CPABORT(message) CALL abort_test(message)\n'
        '#define __LOCATION__ "Cholesky mock"\n'
    )
    objects = []
    for filename in ("mock_cholesky", "mock_offload"):
        obj = directory / (filename + ".o")
        subprocess.run(compiler + flags + include + ["-c", str(HERE / (filename + ".c")),
                                                    "-o", str(obj)], check=True)
        objects.append(str(obj))
    fc = shlex.split(os.environ.get("FC", "gfortran"))
    lapack = shlex.split(os.environ.get("LAPACK_LIBS", "-llapack"))
    configurations = {
        "parallel": (["-D__parallel"], [], True),
        "parallel-no-omp": (["-D__parallel"], [], False),
        "parallel-no-pw": (["-D__parallel"], ["-D__OFFLOAD", "-D__NO_OFFLOAD_PW"], True),
        "dlaf": (["-D__parallel", "-D__DLAF"], [], True),
        "hip": (["-D__parallel", "-D__TEST_HIP"], ["-D__OFFLOAD", "-D__OFFLOAD_HIP"], True),
        "serial": ([], [], True),
    }
    for name, (defines, cdefines, openmp) in configurations.items():
        obj = directory / ("production-" + name + ".o")
        subprocess.run(compiler + flags + include + cdefines + ["-include", str(HERE / "mock_cholesky.h"),
                       "-c", str(build / "src/pw/gpu/pw_gpu_internal.c"), "-o", str(obj)], check=True)
        fflags = ["-cpp", "-ffree-form", "-std=f2008", "-fcheck=all", "-Werror=line-truncation"]
        if openmp:
            fflags += ["-fopenmp"]
        fflags += (["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
                   if sanitize else ["-O2"])
        executable = directory / name
        subprocess.run(fc + fflags + defines + [str(HERE / "mock_cholesky_dependencies.F"),
                       str(directory / "fm/cp_fm_cholesky.F"), str(HERE / "test_cholesky.F"),
                       str(obj), *objects, "-Wl,--export-dynamic", *lapack, *blas,
                       "-lgomp", "-ldl", "-o", str(executable)], cwd=directory, check=True)
        for setting in ("0", "1"):
            result = subprocess.run([str(executable)],
                                    env=environment | {"CP2K_CHOLESKY_FAST_SINGLE_RANK": setting},
                                    text=True, capture_output=True, timeout=120)
            assert result.returncode == 0, (name, setting, result.stdout, result.stderr)
            assert "AddressSanitizer" not in result.stderr and "runtime error:" not in result.stderr, result.stderr
            print(f"PASS Cholesky Fortran {name}, gate={setting}: numerics and dispatch", flush=True)
