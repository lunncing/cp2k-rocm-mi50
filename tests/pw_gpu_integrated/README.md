# Integrated PW GPU memory, XC, and CPU integral tests

Run from the repository root with Python 3.9+, Fypp, a C11 compiler supporting
OpenMP, GNU Fortran, and an LP64 BLAS providing `ddot_` (GCC/GFortran and
OpenBLAS on Linux/WSL work). On Ubuntu the dependencies are available as
`gcc gfortran libblas-dev fypp`:

```sh
python3 tests/pw_gpu_integrated/run_tests.py
python3 tests/pw_gpu_integrated/run_tests.py --sanitize
```

`CC` and `FC` can select compatible C and Fortran compilers. `BLAS_LIBS` supplies
BLAS linker flags (default `-lblas`, for example `-L/path/to/lib -lopenblas`).
The second command enables AddressSanitizer, UndefinedBehaviorSanitizer, and
leak detection. Fortran bounds checks are enabled in both builds.
No CUDA/HIP installation, GPU, MPI,
or complete CP2K checkout is required. Temporary build files stay under this test
directory and are removed automatically.

The tests compile the actual `pw_gpu_internal.c` and production headers, with only
the offload runtime, FFT library, and kernel launchers mocked. The mock tracks every
allocation and free, enforces its own allocation ceiling, checks device pointer
bounds, and rejects freeing memory with pending asynchronous operations. Per-stream
vector clocks model event dependencies and reject unordered accesses across streams,
even though numerical operations execute immediately on the CPU. Its FFT is
an independent direct DFT, used to check the production CPU fallback numerically.

Coverage includes:

- Exact budget boundaries, replacement growth, overflow rejection, unlimited
  allocation, lowering a live limit, current/peak accounting, and reinitialization.
- Automatic environment detection before initialization and during initialization;
  rank 0/rank 1, all four rank variables, precedence, invalid/overflowing values,
  and actual allocation enforcement from a rank override.
- LRU eviction on both memory pressure and cache capacity; map/g-vector accounting,
  reuse, layout changes, invalidation, and protection of the map being extended.
- Every public GPU transform, including fused gradient/divergence/filter,
  one/two-map layouts, distributed FFT layouts, aliased outputs, empty grids,
  fallback at each allocation stage, a single notice, and disabled-fallback failure.
- Dual-stream gradient downloads and divergence uploads, including independent
  transfer/FFT scheduling, buffer reuse dependencies, repeated event recordings,
  and completion of both streams at the public boundary. Serial and dual-buffer
  results are compared byte for byte for one/two-map layouts and aliased divergence
  outputs. Exact serial/dual budgets and a budget one byte short of dual allocation
  exercise GPU-only fallback with host fallback enabled and disabled.
- Optional buffer reuse, growth with pending work on both streams, reclamation
  before required scratch, active-grid protection and LRU eviction, lowering a
  live limit, and stream/event/buffer cleanup across reinitialization.
- Prime and mixed-radix host FFT sizes, no-offload compilation, input keyword and
  C-binding declarations, and the 132-character line limit in the modified Fortran files.
- `pw_cpu_dot_complex_c` linked to real BLAS: deterministic random vectors of
  lengths 0 through 65537, real/imaginary signs, unaligned slices, and aliased
  inputs, compared against an independent long-double complex inner product.
- All 16 generated `pw_integral_ab` variants compiled from the production
  template: default/accurate/standard sums, cell-volume scaling, `just_sum`,
  full/half-space and G=0 corrections, local/distributed reductions, empty local
  grids, strided arrays, and rejection of incompatible grids. Random complex
  vectors are also compared with Fortran `SUM(REAL(CONJG(a)*b))` directly.
  These tests run with offload, without offload, and with `__NO_OFFLOAD_PW`.

The FFT comparisons use a relative/absolute tolerance of `2e-11`; CPU and GPU
FFTs need not be bitwise identical. Dot-product comparisons allow floating-point
summation-order differences using product magnitudes and machine precision;
the C reference check also accounts for vector length. DDOT replaces compensated
summation only for contiguous `c1d/c1d/gs` arrays whose doubled length fits the
BLAS integer range. Strided/oversized arrays retain the original sum paths.

The Fortran harness extracts the actual integral routine and its C binding from
`pw_methods.F` and preprocesses it with Fypp. Minimal grid, timer, reduction, and
accurate-sum stubs replace CP2K modules absent from this snapshot; the C helper
and BLAS are real. The reduction stub checks call counts and applies a known
factor, while the sum stubs check that other variants retain their sum mode.

## Configuration

The memory limit is in MiB (`1024 * 1024` bytes); zero means unconstrained. The
precedence is a valid `CP2K_GPU_MEM_LIMIT_MB_RANK<N>` value, then a valid
`CP2K_GPU_MEM_LIMIT_MB` value, then the Fortran/C setter (default zero). A rank
override of zero explicitly selects unlimited memory. Invalid, negative, or
overflowing values are ignored. Rank is the first valid value from
`OMPI_COMM_WORLD_RANK`, `PMI_RANK`, `SLURM_PROCID`, `MPI_LOCALRANKID`, or zero if none
is present. Environment settings are captured once and override later input setters.

Host fallback defaults to enabled. `CP2K_GPU_HOST_FALLBACK=0` or `FALSE` disables it;
the existing case-insensitive `no`/`off` switches also work. A budget failure with
fallback disabled reports an error and terminates before executing a transform.
With fallback enabled, the operation completes on the CPU and reports one notice
per process. The portable mixed-radix host FFT requires no extra FFT library;
large prime transform lengths can be slower because their radix uses a direct DFT.

`MAX_GPU_MEMORY_MB` and `HOST_FALLBACK` are declared under `GLOBAL%GRID`.
`input_cp2k_global:read_global_section(global_section)` reads a parsed GLOBAL
section and calls the public setters in `pw_gpu`. This source snapshot has no
startup GLOBAL reader/caller to connect; the environment path initializes itself
inside C regardless of whether the input hook is called. No files outside the
specified implementation files, necessary GPU header, and this test directory
are changed.

The accounting covers the two shared work buffers, the optional real/complex
secondary pair, the legacy map, compact G-space scratch, and every cached map and
g-vector allocation. Opaque FFT-library
workspaces and allocations from other CP2K components are outside this budget.
These mocks do not replace a full CP2K build or CUDA/HIP integration run.

A scan of all Fortran sources also found two existing out-of-scope violations:
`src/xc/xc_rho_set_types.F:407` (187 characters) and `:463` (227 characters).
They match the original checkout and are left unchanged under the file-scope restriction.
