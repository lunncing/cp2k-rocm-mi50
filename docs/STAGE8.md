# Stage 8: UKS PBE, diagonalization, and reciprocal-space mixing

This overlay extends the Stage 7 implementation in the isolated
`feat/stage8-uks-elpa-vram` worktree.

## Runtime controls

| Setting | Default | Behavior |
| --- | --- | --- |
| `CP2K_PW_GPU_UKS` | `1` | Enables standalone spin-polarized PBE and the dual-spin PW paths. |
| `CP2K_PW_GPU_XC_FULL_PIPELINE` | `1` | Enables the fused UKS XC pipeline. Setting it to `0` retains the existing XC orchestration. |
| `CP2K_ELPA_GPU_N_MIN` | `12288` | Minimum matrix order for AUTO to select the AMD GPU kernel. Explicit AMD_GPU bypasses this threshold. |
| `CP2K_CHOLESKY_FAST_SINGLE_RANK` | `1` | Enables local LAPACK/BLAS and optional HIP triangular operations on one MPI rank and a 1 x 1 process grid, outside an active OpenMP team. Set to `0` to retain the distributed routines. |

The existing memory budget, host fallback, residency, lazy potential, natural
ordering, paired FFT, and density z-derivative controls continue to apply.
Invalid ELPA threshold values use the default. A threshold of zero removes
the AUTO size restriction.

The UKS fused path requires one active `PBE` functional with the original PBE
parametrization, PW derivatives, no density smoothing, no adiabatic rescaling,
and no virial, triplet, weight, or spatial energy-output request. revPBE,
PBEsol, additional functionals, and unsupported layouts retain the original
implementation. Standalone LSD offload handles derivative orders -1, 0, and 1;
higher derivatives and other parametrizations stay in Fortran.

## Device lifetime

The actual Stage 7 density routine in this overlay stores density results in
the XC allocations. On entry of the second spin, Stage 8 copies the alpha
reciprocal and real densities into persistent, budgeted UKS storage before
reusing those allocations. The beta real density also occupies persistent
UKS storage, so Hartree's shared FFT buffers cannot overwrite it.

The two z derivatives share the existing upper XC allocation: alpha occupies
its first real grid and beta its second. The RKS density layout remains
unchanged. The dedicated UKS allocations total six real grids: four for
`uks_density_dev` and two for `uks_xc_dev`. XC consumes each spin's spectrum
through the shared reciprocal scratch before overwriting it with gradients.
The two real densities stay in the upper half of `uks_density_dev`; the lower
halves of the density and XC allocations hold the x/y gradients, and
`uks_xc_dev` holds the z gradients. Once the z derivatives have been consumed,
the upper XC allocation holds the local density derivatives. Energy and its
reduction reuse the shared FFT scratch. The cutoff sums the spin densities
inside the kernel, with no stored total-density grid.

The final alpha potential occupies the first grid of `uks_density_dev`, and
the beta potential occupies the first grid of `xc_dev`. This removes twelve
real grids from the previous UKS workspace (about 5.9 GiB for 66.4 million
points). The shared FFT and XC allocations retain their existing sizes.

Both potentials can remain device-authoritative. Vtot preparation and host
skipping track the selected spin, and the first Hartree-potential upload can
be cached for the second spin. Charge integration also selects the density by
host identity, so calculating the beta charge downloads only a scalar.
Reclamation synchronizes live results before freeing their storage, and every
explicit device allocation participates in the existing accounting.

When host fallback is enabled, mandatory PW allocations also check the
driver's free VRAM, including allocations by FFT libraries and other processes.
They preserve `CP2K_PW_GPU_RES_RESERVE_MB` (1024 MiB in the Stage 7 runtime);
an explicit zero uses a 256 MiB floor for this check. Reclamation tries optional
double buffers, unpinned owned slots, unprotected UKS/XC storage, and then the
oldest unprotected grid cache. Insufficient headroom returns to the host path
before allocation. UKS preparation protects its resident inputs throughout
allocation and grid setup. Unknown device memory information skips the check,
as does `CP2K_GPU_HOST_FALLBACK=0`, which retains strict budget/malloc behavior.

The public `pw_gpu_xc_full_pipeline_lsd_c` entry accepts separate G-vector
components and one G-to-FFT map. The configured entry used by the Fortran
wrapper accepts CP2K's interleaved G vectors, grid identity, and one or two maps
without constructing new host arrays each iteration.

Hartree supports both spin counts, DDAPC prefixes and corrections, and the
direct reciprocal-grid Poisson path without dielectric screening. Failure
returns control to the existing host orchestration.

## ELPA selection

AMD setup prefers `amd-gpu` and uses `gpu` only when the former is unavailable.
GPU setup runs once per new handle. Unsupported block sizes or failed GPU
setup select AVX512_BLOCK4, AVX2_BLOCK2, or GENERIC according to ELPA support.
The cached handle includes matrix order, block size, requested and resolved
kernel, communicator, local layout, solver, and thread count.

Measurements on Ryzen 9 9950X + MI50 over matrix orders 128 through 8320
showed `AVX512_BLOCK4` running 1.6–4.2 times faster than
`ELPA_2STAGE_REAL_AMD_GPU`. The calibrated AUTO default is therefore 12288,
above the measured range; this is a conservative selection policy, not a
measured GPU crossover. `CP2K_ELPA_GPU_N_MIN=0` removes the size gate and
explicit `ELPA_KERNEL AMD_GPU` bypasses it.

## Section III: single-rank Cholesky and triangular operations

In the measured `Broy./Diag.` runs, Cholesky decomposition, reduction and
restoration accounted for 38–56% of `qs_scf_new_mos`. Parallel builds previously
used ScaLAPACK/PBLAS even with one MPI rank, invoking small block operations
with `OPENBLAS_NUM_THREADS=1`.

The fast path requires one MPI rank, a 1 x 1 process grid, one currently
executing OpenMP thread, and contiguous local storage with sufficient rows
and columns for every operand. Padded leading dimensions are supported.
Decomposition, solve and inversion call `dpotrf`, `dpotrs` and `dpotri`
directly. Explicit DLA-Future selection retains its decomposition and inversion
dispatch. Multiple ranks or an active OpenMP team retain the distributed path.
Builds without OpenMP use a thread-count default of one.

OpenBLAS thread controls are resolved dynamically. Local LAPACK and triangular
BLAS calls temporarily use `omp_get_max_threads()` and restore the previous
OpenBLAS count, including when LAPACK reports an error. Missing OpenBLAS symbols
leave the library's thread settings unchanged. Reduction with `itype=1` and
order at least 512 mirrors the authoritative upper triangle and uses two
`dtrsm` calls to compute `inv(U^T) A inv(U)`. The measured CPU improvement over
`dsygst` was 4.3 times at order 6656. Smaller reductions and other `itype`
values use `dsygst` with one OpenBLAS thread.

When the PW HIP runtime is initialized, restore and reduction first try
dynamically resolved rocBLAS routines on the PW stream. Restore requires at
least 512 rows and one column, plus either at least 256 columns for a solve
or at least 512 x 512 elements. Reduction requires order at least 512.
One rocBLAS handle is cached and destroyed by PW finalization. Packed device
storage uses `bdim^2 + m*n` doubles for a solve and `bdim^2 + 2*m*n` for a
multiply, where `bdim` is the triangular factor's order. Allocation checks
the PW budget and device headroom, uses fallible HIP allocation, and contributes
to PW memory accounting. Host staging preserves caller data until all device
work and downloads succeed. Missing rocBLAS, insufficient memory, or a HIP
or rocBLAS failure returns to the CPU path. The C bindings also exist in CPU
and disabled-PW builds; no FM dependency on the `pw_gpu` Fortran module is needed.

Restore maps real-BLAS `transa='C'/'c'` to transpose for GPU dispatch, just
like `T/t`. The existing `N/T` mappings and CPU BLAS arguments are unchanged.
The Fortran entry retains its existing assertion that the first character
of `pos` is uppercase `L` or `R`; lowercase sides cannot silently reach a
different GPU operation.

## Reciprocal-space mixing

`src/qs_gspace_mixing.F` was absent from the Stage 7 overlay and is based on
the upstream `v2026.2` file. Spin/charge conversions now use direct parallel
loops. Independent Kerker, Pulay, and Broyden updates run in parallel.
Scalar norms retain their original accumulation order; Pulay/Broyden matrix
entries parallelize over independent history indices, with each entry's
G-vector sum remaining serial. MPI reductions remain outside OpenMP regions.

## Validation

Run from the worktree root in WSL:

```sh
python3 tests/pw_gpu_integrated/generate_pbe_kernel.py --check
python3 tests/pw_gpu_integrated/run_tests.py --sanitize
python3 tests/pw_gpu_residency/run_tests.py
python3 tests/elpa_gpu/run_tests.py
```

The integrated numerical tests require BLAS and LAPACK development libraries
(`libblas-dev` and `liblapack-dev` on Ubuntu). `BLAS_LIBS` and `LAPACK_LIBS`
can override their link flags. For a focused Cholesky run, use
`python3 tests/pw_gpu_integrated/run_tests.py --sanitize --cholesky-only`.

The suites compile production C with ASan/UBSan, emulate production CUDA/HIP
kernel bodies, and compare LSD energy and derivatives against the independent
compiled Fortran calculation. Pipeline comparisons use a Fortran DFT reference.
Residency tests cover spin preservation, natural and sorted order, spherical
and conjugate maps, cached z derivatives, Hartree/DDAPC, both Vtot slots,
lazy/eager operation, and memory reclamation. Compiled bindings cover offload,
CPU, and disabled-PW builds. Mixing checks compare all 20 parallel loops
bitwise against serial execution and separately verify spin conversion
arithmetic. ELPA mocks check setup, caching, fallback order, and thresholds.
Cholesky tests cover all solve/multiply, left/right and normal/transpose
combinations, padded leading dimensions, aliased input/output, reduction
against independent references, gate settings, and handle lifetime. Injected
allocation, upload, download, synchronization and rocBLAS failures verify
unchanged host inputs and complete temporary cleanup. Compiled Fortran tests
exercise local and distributed dispatch, DLA-Future, CPU, HIP, disabled-PW and
OpenMP-disabled builds, including thread-count restoration.

The Cholesky review regressions call production Fortran restore with `C/c`
at order 512 for solve/multiply and left/right operations, compare against
CPU `dtrsm`/`dtrmm`, and assert that the HIP mock dispatch completes without
CPU fallback. Order-8 and gate-disabled cases also check the CPU paths.
Two order-512, `itype=1` reductions inject failure at the first or second
mock rocBLAS solve. They verify that the failure is consumed, the Fortran
symmetric expansion and two-`dtrsm` fallback match the upper triangle from
`dsygst` at the existing `2e-11` tolerance, leading-dimension padding survives,
and the temporary OpenBLAS thread count is restored. The input lower
triangle contains the padding sentinel to exercise symmetric expansion.

These checks do not establish performance or numerical behavior on physical
AMD hardware, or replace a full CP2K build and molecular regression run.

Verification on November 14, 2026: all four commands completed successfully
in WSL. The integrated suite passed 89 C mock cases, the compiled Fortran
comparisons, and all 20 mixing-loop comparisons, plus 25 Cholesky C
configurations and 12 compiled Cholesky Fortran runs. Cholesky C builds use
`-std=c11 -Wall -Wextra -Werror -Wpedantic`; the expanded integrated suite
passed with ASan/UBSan. The residency suite passed
the dual-spin lifecycle, beta-allocation failure recovery, charge integration,
and compiled periodic/DDAPC orchestration checks. ELPA passed 192 capability
and threshold combinations, plus compilation without ELPA. The final runs
reported no compiler warnings or sanitizer errors. `git diff --check` was
clean, and all 33 repository Fortran files had lines of at most 132 characters.
