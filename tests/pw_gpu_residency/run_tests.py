#!/usr/bin/env python3
"""PW GPU residency tests against the checked CPU mock (no GPU required).

Builds the production src/pw/gpu C sources with ASan/UBSan and runs
  * Hermes's C cases from ../pw_gpu_integrated (must keep passing),
  * the emulated .cu kernel bodies (Hermes's checks + the residency kernels),
  * the residency cases: every fused path bit for bit against the original
    call sequence through the same mock FFT, with 1 and 3 OpenMP threads.
Usage (from the CP2K source root):  python3 tests/pw_gpu_residency/run_tests.py
"""
import os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HERMES = HERE.parent / "pw_gpu_integrated"
if not (HERMES / "mock_offload.c").is_file():
    sys.exit(f"missing {HERMES} (Hermes's pw_gpu_integrated test kit, updated version)")
sys.path.insert(0, str(HERMES))
sys.dont_write_bytecode = True
import run_tests as hermes  # noqa: E402
from fortran_tests import run_fortran_tests  # noqa: E402

RES_CASES = ["density", "density_rejects", "potential", "hartree", "chain", "memory",
             "borrowed", "samples", "verify"]


def main():
    hermes.source_checks()
    cc = os.environ.get("CC", "gcc").split()
    cxx = os.environ.get("CXX", "g++").split()
    blas = os.environ.get("BLAS_LIBS", "-lblas").split()
    flags = ["-std=c11", "-Wall", "-Wextra", "-Werror", "-Wpedantic", "-fopenmp", "-ffp-contract=off",
             "-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    with tempfile.TemporaryDirectory(prefix=".build-", dir=HERE) as tmp:
        build = Path(tmp)
        gpu = build / "src/pw/gpu"
        gpu.mkdir(parents=True)
        for path in (ROOT / "src/pw/gpu").iterdir():
            if path.suffix in (".h", ".c"):
                shutil.copy2(path, gpu)
        off = build / "src/offload"
        off.mkdir()
        for name in ("offload_runtime.h", "offload_fft.h", "offload_library.h"):
            (off / name).write_text('#include "mock_offload.h"\n')
        inc = ["-I", str(HERMES), "-I", str(gpu)]
        exes = {}
        for test, src in (("test_pw_gpu", HERMES / "test_pw_gpu.c"), ("test_res", HERE / "test_res.c")):
            exes[test] = build / test
            subprocess.run(cc + flags + inc + ["-D__OFFLOAD", str(src), str(HERMES / "mock_offload.c"),
                           *blas, "-lm", "-o", str(exes[test])], check=True)
        for defines in ([], ["-D__OFFLOAD", "-D__NO_OFFLOAD_PW"]):
            subprocess.run(cc + flags + inc + defines + ["-c", str(gpu / "pw_gpu_internal.c"),
                           "-o", str(build / "x.o")], check=True)
        env = {k: v for k, v in os.environ.items() if k not in hermes.RANK_NAMES and
               not k.startswith(("CP2K_GPU_", "CP2K_PW_GPU_"))}
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                   UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
        source = (ROOT / "src/pw/gpu/pw_gpu_kernels.cu").read_text()
        source, count = re.subn(r"(\w+)<<<(.*?)>>>\(", r"emulate(\1, \2, ", source, flags=re.S)
        assert count == 20, count
        prelude = (HERMES / "kernel_body_tests.py").read_text().split("prelude = '''", 1)[1].split("'''", 1)[0]
        body = (HERMES / "test_kernel_bodies.cpp").read_text()
        body = body.replace("int main() {", "static void hermes_main() {", 1)
        body += (HERE / "test_res_kernels.cpp").read_text()
        path = gpu / "emulated_kernels.cpp"
        path.write_text(prelude + source + body)
        cxxflags = [f for f in flags if f != "-std=c11"] + ["-std=c++17"]
        subprocess.run(cxx + cxxflags + inc + ["-D__OFFLOAD", str(path), "-o", str(build / "kern")], check=True)
        subprocess.run([str(build / "kern")], env=env, check=True, timeout=60)
        print("PASS emulated kernel bodies (Hermes + residency)")
        n = 0
        for name, args, overrides, status, notices in hermes.cases():
            r = subprocess.run([str(exes["test_pw_gpu"]), *args], env=env | overrides, text=True,
                               capture_output=True, timeout=60)
            if status == -1:
                assert r.returncode != 0 and "host fallback is disabled" in r.stderr, (name, r)
            else:
                assert r.returncode == status, (name, r.stdout, r.stderr)
                assert r.stdout.strip() == "PASS", (name, r)
            assert r.stderr.count("falling back to host.") == notices, (name, r.stderr)
            assert "AddressSanitizer" not in r.stderr and "runtime error:" not in r.stderr, r.stderr
            n += 1
        print(f"PASS {n} Hermes C mock cases")
        cases = [(name, {"CP2K_PW_GPU_RES_LAZY_DENSITY": "0"}) for name in RES_CASES]
        cases += [(name, {}) for name in ("lazy_density", "integrate", "chain", "lazy_vtot")]
        for name, overrides in cases:
            for threads in ("1", "3"):
                r = subprocess.run([str(exes["test_res"]), name],
                                   env=env | overrides | {"OMP_NUM_THREADS": threads},
                                   text=True, capture_output=True, timeout=120)
                assert r.returncode == 0 and r.stdout.strip() == "PASS", (name, r.stdout, r.stderr)
                assert "AddressSanitizer" not in r.stderr and "runtime error:" not in r.stderr, r.stderr
            print(f"PASS residency {name} ({'eager' if overrides else 'lazy'}, OMP=1/3)")
        run_fortran_tests(build, cc, flags, inc, blas, env)


if __name__ == "__main__":
    main()
