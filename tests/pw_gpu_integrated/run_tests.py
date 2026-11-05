#!/usr/bin/env python3
"""Compile the production PW GPU code against a checked CPU mock (no GPU needed)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
from integral_tests import run_integral_tests
from grid_ops_tests import run_grid_ops_tests
from kernel_body_tests import run_kernel_body_tests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RANK_NAMES = ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID", "MPI_LOCALRANKID")


def source_checks() -> None:
    for name in ("src/input_cp2k_global.F", "src/pw/pw_gpu.F", "src/pw/pw_methods.F",
                 "src/cp_ddapc_util.F", "src/xc/xc_pbe.F", "tests/pw_gpu_integrated/test_grid_ops.F",
                 "tests/pw_gpu_integrated/test_pw_integral.F"):
        lines = (ROOT / name).read_text().splitlines()
        long_lines = [(i, len(line)) for i, line in enumerate(lines, 1) if len(line) > 132]
        assert not long_lines, (name, long_lines)
        print(f"PASS {name}: maximum Fortran line length {max(map(len, lines))}")
    source = (ROOT / "src/input_cp2k_global.F").read_text()
    grid = source.split("SUBROUTINE create_grid_section(section)", 1)[1]
    for name, default, kind in (("MAX_GPU_MEMORY_MB", "default_i_val=0", "integer_t"),
                                ("HOST_FALLBACK", "default_l_val=.TRUE.", "logical_t")):
        keyword = grid.split(f'name="{name}"', 1)[1].split("CALL section_add_keyword", 1)[0]
        assert default in keyword and f"type_of_var={kind}" in keyword
    reader = source.split("SUBROUTINE read_global_section(global_section)", 1)[1]
    reader = reader.split("END SUBROUTINE", 1)[0]
    assert '"GRID%MAX_GPU_MEMORY_MB"' in reader and '"GRID%HOST_FALLBACK"' in reader
    assert "CALL pw_gpu_set_memory_limit(max_gpu_memory_mb)" in reader
    assert "CALL pw_gpu_set_host_fallback(host_fallback)" in reader
    wrapper = (ROOT / "src/pw/pw_gpu.F").read_text()
    for name in ("pw_gpu_set_memory_limit", "pw_gpu_set_host_fallback", "pw_gpu_get_allocated_bytes"):
        assert re.search(r"PUBLIC ::[^\n]*\b" + name + r"\b", wrapper)
        assert f'BIND(C, name="{name}")' in wrapper
    assert "INTEGER(KIND=C_SIZE_T), INTENT(OUT)" in wrapper
    print("PASS input keywords, configuration reader, and interoperable Fortran bindings")


def cases():
    yield "grid kernels bitwise", ["grid_ops"], {}, 0, 0
    yield "new operations exact budgets and overflow", ["new_limits"], {}, 0, 1
    for op in range(6):
        yield f"new operation {op} disabled fallback", ["new_disabled", str(op)], {}, -1, 0
    for name in ("limits", "eviction", "fallback", "late_fallback", "empty", "fft_sizes", "environment_once",
                 "dual_stream", "dual_buffer_memory"):
        yield name, [name], {}, 0, int(name in ("limits", "fallback", "late_fallback"))
    yield "fallback disabled by setter", ["disabled"], {}, -1, 0
    yield "fallback disabled by environment", ["disabled"], {"CP2K_GPU_HOST_FALLBACK": "FALSE"}, -1, 0
    yield "dual stream with host fallback disabled", ["dual_stream"], {"CP2K_GPU_HOST_FALLBACK": "FALSE"}, 0, 0
    yield "automatic rank budget enforcement", ["environment_limit"], {
        "OMPI_COMM_WORLD_RANK": "1", "CP2K_GPU_MEM_LIMIT_MB": "3", "CP2K_GPU_MEM_LIMIT_MB_RANK1": "1"}, 0, 1
    # A setting is read exactly once, on either first query or initialization.
    environments = [
        ("defaults", {}, 0, 1),
        ("global", {"CP2K_GPU_MEM_LIMIT_MB": "3"}, 3, 1),
        ("implicit rank zero", {"CP2K_GPU_MEM_LIMIT_MB": "9", "CP2K_GPU_MEM_LIMIT_MB_RANK0": "2"}, 2, 1),
        ("rank one", {"CP2K_GPU_MEM_LIMIT_MB": "9", "CP2K_GPU_MEM_LIMIT_MB_RANK0": "2",
                       "CP2K_GPU_MEM_LIMIT_MB_RANK1": "4", "OMPI_COMM_WORLD_RANK": "1"}, 4, 1),
        ("rank precedence", {"CP2K_GPU_MEM_LIMIT_MB_RANK0": "2", "CP2K_GPU_MEM_LIMIT_MB_RANK1": "4",
                             "OMPI_COMM_WORLD_RANK": "0", "PMI_RANK": "1"}, 2, 1),
        ("invalid rank", {"OMPI_COMM_WORLD_RANK": "-1", "PMI_RANK": "1",
                          "CP2K_GPU_MEM_LIMIT_MB_RANK1": "4"}, 4, 1),
        ("oversized rank", {"OMPI_COMM_WORLD_RANK": "99999999999999999999", "SLURM_PROCID": "1",
                            "CP2K_GPU_MEM_LIMIT_MB_RANK1": "4"}, 4, 1),
        ("missing rank override", {"PMI_RANK": "1", "CP2K_GPU_MEM_LIMIT_MB": "6",
                                   "CP2K_GPU_MEM_LIMIT_MB_RANK0": "4"}, 6, 1),
        ("rank zero unlimited", {"CP2K_GPU_MEM_LIMIT_MB": "9", "CP2K_GPU_MEM_LIMIT_MB_RANK0": "0"}, 0, 1),
        ("invalid rank override", {"CP2K_GPU_MEM_LIMIT_MB": "3", "CP2K_GPU_MEM_LIMIT_MB_RANK0": "oops"}, 3, 1),
        ("padded integer", {"CP2K_GPU_MEM_LIMIT_MB": " 5 "}, 5, 1),
    ]
    for rank_name in RANK_NAMES:
        environments.append((rank_name, {rank_name: "1", "CP2K_GPU_MEM_LIMIT_MB_RANK1": "7"}, 7, 1))
    for invalid in ("", "-2", "no", "3oops", "1.5", "18446744073709551616", "17592186044416"):
        environments.append((f"invalid limit {invalid!r}", {"CP2K_GPU_MEM_LIMIT_MB": invalid}, 0, 1))
    for value, enabled in (("0", 0), ("FALSE", 0), ("false", 0), ("OFF", 0), ("1", 1), ("TRUE", 1), ("", 1)):
        environments.append((f"fallback {value!r}", {"CP2K_GPU_HOST_FALLBACK": value}, 0, enabled))
    for name, env, mb, enabled in environments:
        for first_init in (0, 1):
            yield f"{name}, {'init' if first_init else 'query'}", [
                "environment", str(mb), str(enabled), str(first_init), "1"], env, 0, 0
    # Map cache disabled must retain both normal and fallback transformations.
    yield "legacy maps numerical fallback", ["fallback"], {"CP2K_PW_GPU_MAP_CACHE": "0"}, 0, 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize", action="store_true", help="enable ASan and UBSan")
    args = parser.parse_args()
    if os.name == "posix":
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    source_checks()
    compiler = shlex.split(os.environ.get("CC", "cc"))
    blas = shlex.split(os.environ.get("BLAS_LIBS", "-lblas"))
    flags = ["-std=c11", "-Wall", "-Wextra", "-Werror", "-Wpedantic", "-fopenmp", "-ffp-contract=off"]
    flags += ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if args.sanitize else ["-O2"]
    with tempfile.TemporaryDirectory(prefix=".build-", dir=HERE) as temporary:
        build = Path(temporary)
        gpu = build / "src/pw/gpu"
        gpu.mkdir(parents=True)
        # Preserve the production includes and conditional compilation exactly;
        # supply only the absent offload backend, in an isolated test build tree.
        for path in (ROOT / "src/pw/gpu").iterdir():
            if path.suffix in (".h", ".c"):
                shutil.copy2(path, gpu)
        offload = build / "src/offload"
        offload.mkdir()
        for name in ("offload_runtime.h", "offload_fft.h", "offload_library.h"):
            (offload / name).write_text('#include "mock_offload.h"\n')
        include = ["-I", str(HERE), "-I", str(gpu)]
        exe = build / "test_pw_gpu"
        command = compiler + flags + include + ["-D__OFFLOAD", str(HERE / "test_pw_gpu.c"),
                    str(HERE / "mock_offload.c"), *blas, "-lm", "-o", str(exe)]
        subprocess.run(command, check=True)
        for defines in ([], ["-D__OFFLOAD", "-D__NO_OFFLOAD_PW"]):
            subprocess.run(compiler + flags + include + defines + ["-c", str(gpu / "pw_gpu_internal.c"),
                           "-o", str(build / "no_offload.o")], check=True)
        clean_env = {k: v for k, v in os.environ.items() if k not in RANK_NAMES and
                     not k.startswith(("CP2K_GPU_", "CP2K_PW_GPU_"))}
        clean_env["OMP_NUM_THREADS"] = "1"
        clean_env["OPENBLAS_NUM_THREADS"] = "1"
        clean_env["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
        clean_env["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
        run_kernel_body_tests(build, flags, include, clean_env)
        count = 0
        for name, arguments, overrides, status, notices in cases():
            result = subprocess.run([str(exe), *arguments], env=clean_env | overrides,
                                    text=True, capture_output=True, timeout=60)
            if status == -1:
                assert result.returncode != 0 and "host fallback is disabled" in result.stderr, (name, result)
            else:
                assert result.returncode == status, (name, result.stdout, result.stderr)
                assert result.stdout.strip() == "PASS", (name, result)
            assert result.stderr.count("falling back to host.") == notices, (name, result.stderr)
            assert "AddressSanitizer" not in result.stderr and "runtime error:" not in result.stderr, result.stderr
            print(f"PASS {name}")
            count += 1
        print(f"{count} C mock cases passed; no-offload variants compiled; "
              f"{'ASan/UBSan enabled' if args.sanitize else 'optimized build'}.")
        run_integral_tests(build, compiler, flags, include, blas, clean_env, args.sanitize)
        run_grid_ops_tests(build, compiler, flags, include, blas, clean_env, args.sanitize)


if __name__ == "__main__":
    main()
