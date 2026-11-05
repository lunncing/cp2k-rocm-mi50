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
check the cache's polymorphic declaration.

CPU, HIP, CUDA, and OpenCL builds each exercise all eight combinations of generic
GPU, AMD GPU, and OpenMP option support. The HIP build also enables IEEE exception
handling. Compilation without ELPA is checked separately.

These tests validate wrapper behavior; real MPI collectives, ROCm performance,
GPU numerical results, and ELPA integration require a full CP2K installation.
