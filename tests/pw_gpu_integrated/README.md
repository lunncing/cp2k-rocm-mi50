# Integrated PW GPU grid, Hartree, PBE, memory, XC, and CPU integral tests

Run from the repository root with Python 3.9+, Fypp, C11 and C++17 compilers supporting
OpenMP, GNU Fortran, and an LP64 BLAS providing `ddot_` (GCC/G++/GFortran and
OpenBLAS on Linux/WSL work). On Ubuntu the dependencies are available as
`gcc g++ gfortran libblas-dev fypp`:

```sh
python3 tests/pw_gpu_integrated/run_tests.py
python3 tests/pw_gpu_integrated/run_tests.py --sanitize
```

`CC`, `CXX`, and `FC` can select compatible C, C++, and Fortran compilers. `BLAS_LIBS` supplies
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

- All 16 generated copy and AXPY variants compared bit for bit against the
  original Fortran branches: real/complex conversions, contiguous 1D/3D data,
  spherical padding/truncation, both reference-grid relations, gather/scatter,
  arbitrary alpha/beta, strided arrays, aliasing, empty arrays, NaNs and signed zero.
- Hartree compared to the compiled original atom/Gaussian loops, with complex
  coefficients, multiple atoms and radii, G=0 present/absent, the GCUT prefix,
  empty G ranges, and exact CPU fallback. No atom reduction reorders the sum.
- PBE, revPBE and PBEsol energy and first derivatives compared to compiled
  `pbe_lda_calc` at `1e-12` relative/absolute tolerance. Covers density cutoff
  neighbors, nonpositive/NaN densities, zero gradients, independent exchange and
  correlation scaling, accumulated outputs, and aliased unused derivative arrays.
  Derivative orders 2/3 retain the original CPU routine. OpenMP teams of 1 and 3
  execute exactly one GPU attempt via SINGLE/COPYPRIVATE.
- All new C entry points at exact and one-byte-short memory limits; size overflow,
  untouched outputs on failed attempts, disabled fallback and allocation cleanup.
- Production `.cu` bodies and launchers compiled as C++ with CPU launch emulation:
  empty launches, block boundaries/tails, and grid strides. Only launch syntax and
  device annotations are replaced. The compiled Fortran comparisons independently
  check the shared scalar arithmetic used by these kernels and the runtime mock.
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
inside C regardless of whether the input hook is called.

The new C entry points take host pointers and stage each operation in the existing
compact scratch allocation. The kernels operate on device pointers. Every required
buffer is reserved before uploads or output writes; a failed attempt returns zero
so the caller executes its unchanged CPU branch. Grid calls in an OpenMP team and
AXPY aliases that require CPU scaling order also retain the CPU branch. Nonfinite
complex scaling retains the compiler's CPU handling of NaN/Inf cross terms. The one-based
`gidx` maps are uploaded per call, independently of the zero-based FFT map cache;
scatter maps must be injective, as required by the existing CPU parallel loops.

The accounting covers the two shared work buffers, the optional real/complex
secondary pair, the legacy map, compact G-space scratch, and every cached map and
g-vector allocation. Opaque FFT-library
workspaces and allocations from other CP2K components are outside this budget.
These mocks do not replace a full CP2K build or CUDA/HIP integration run.

## Source provenance and PBE maintenance

The supplied snapshot omitted the specification's two callers. `src/cp_ddapc_util.F`
and `src/xc/xc_pbe.F` were restored from CP2K upstream commit
[`37370f362787e9d04d7644023de78707537d81e5`](https://github.com/cp2k/cp2k/tree/37370f362787e9d04d7644023de78707537d81e5)
before adding the GPU calls. Their original CPU routines remain the numerical
references in the tests. Other dependencies of these modules are stubbed, so this
test suite does not constitute a full CP2K build.

`pw_gpu_pbe.h` is generated from the energy/first-derivative expression trees of
`pbe_lda_calc`. Regenerate it with `python3 tests/pw_gpu_integrated/generate_pbe_kernel.py`
when updating those expressions; the suite rejects a stale header. Explicitly
rounded multiply/add operations prevent contraction on CUDA, and HIP/Clang disables
contraction in the element functions. Transcendental functions use double precision;
their hardware results still require verification on a CUDA/HIP system.

A scan of all Fortran sources also found two existing out-of-scope violations:
`src/xc/xc_rho_set_types.F:407` (187 characters) and `:463` (227 characters).
They match the original checkout and are left unchanged under the file-scope restriction.
