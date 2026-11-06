# ELPA GPU and persistent handle regression tests

Run from the repository root with Python 3.9+ and GNU Fortran with OpenMP support:

```sh
python3 tests/elpa_gpu/run_tests.py
```

`FC` selects the Fortran compiler. The runner builds the complete production
`src/fm/cp_fm_elpa.F` against small CP2K and ELPA mocks because this source snapshot
does not include the full CP2K build or ELPA. Temporary build files are removed
automatically. Fortran bounds checks and the 132-character line limit are enforced.

The tests cover persistent allocation/setup reuse, invalidation for every layout
and solver input, a simulated remote-rank cache miss, communicator retention and
release ordering, finalization and reinitialization, GPU backend changes, and QR
diagnostics followed by another solve. ELPA's abstract base type is modeled to
check the cache's polymorphic declaration. Explicit CPU selection must leave all
GPU flags disabled, including in HIP builds and after AUTO selected a GPU.
Explicit AMD selection probes both `gpu` and `amd-gpu` before setting the kernel;
OpenMP threads are set only when that option is supported.

The runner compiles the complete production Fortran module with `__ELPA` in four
build variants. Each runs masks 0–7: bit 0 enables mock `gpu` support, bit 1
`amd-gpu`, and bit 2 `omp_threads`, covering all eight capability combinations.

| Build variant | Additional preprocessor definitions | Mock runs |
|---|---|---:|
| CPU | None | 8 |
| HIP | `__OFFLOAD_HIP`, `__HAS_IEEE_EXCEPTIONS` | 8 |
| CUDA | `__OFFLOAD_CUDA` | 8 |
| OpenCL | `__OFFLOAD_OPENCL` | 8 |

This is **32 mock configurations plus a separate compilation without ELPA**.
The HIP variant includes the IEEE exception-handling path. The runner does not
add a separate `__NO_OFFLOAD_ELPA` build.

`test_elpa.F` is a byte-for-byte copy of the accepted
[Stage 6 r3 test](../../releases/stage6-r3/overrides/source/tests/elpa_gpu/test_elpa.F).
It aligns the root suite with the wrapper already included in that release;
the runner, mocks, and production wrapper are unchanged by this synchronization.

These are compile-time backend variants using CPU-hosted mocks, with no real GPU
or MPI execution. They validate wrapper control flow and returned mock values,
not ELPA numerical accuracy, GPU runtime correctness, or performance. Real ELPA
integration, multi-rank collectives, and diagonalization comparisons require a
full CP2K installation. See the [ELPA optimization guide](../../docs/ELPA-OPTIMIZATION.md)
for activation and release scope.
