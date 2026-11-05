# Specification: ELPA ROCm GPU Acceleration & Persistent Handle Caching

## 1. Background and Motivation
When users compute metallic systems, transition-metal interfaces (e.g. PtRu/C), or difficult-to-converge electronic structures where OT (Orbital Transformation) struggles, CP2K switches to traditional diagonalization (`DIAGONALIZATION`) with Broyden density mixing (`METHOD BROYDEN_MIXING`).
Under diagonalization, the Kohn-Sham eigensolver $H C = S C E$ is called at every SCF iteration.

In CP2K's `src/fm/cp_fm_elpa.F`:
1. GPU kernel selection only sets `amd-gpu=1` when `elpa_kernel` is strictly `ELPA_2STAGE_REAL_AMD_GPU`. In ELPA 2026.02, modern ELPA uses both generic `gpu=1` and vendor-specific `amd-gpu=1`. We should probe `can_set` and enable GPU acceleration robustly.
2. Every call to `cp_fm_diag_elpa_base` calls `elpa_allocate()` and `elpa_deallocate()`. This repeatedly tears down and recreates HIP streams, rocBLAS/hipSOLVER contexts, and device memory pools on every SCF step, adding significant latency.
3. Caching the ELPA handle across SCF steps when dimensions $(n, n_{\text{blk}}, n_{\text{rows}}, n_{\text{cols}})$ and communicator do not change avoids this driver overhead.

## 2. Implementation Requirements in `src/fm/cp_fm_elpa.F`
1. **Robust GPU Flag Probing**:
   In `cp_fm_diag_elpa_base`:
   - If `elpa_kernel == ELPA_2STAGE_REAL_AMD_GPU` or `defined(__OFFLOAD_HIP)`:
     - Check `elpa_obj%can_set("gpu", 1) == ELPA_OK` -> set `gpu=1`.
     - Check `elpa_obj%can_set("amd-gpu", 1) == ELPA_OK` -> set `amd-gpu=1`.
2. **Persistent Handle Caching**:
   - Maintain static/module variables:
     `TYPE(elpa_t), POINTER, SAVE :: cached_elpa_obj => NULL()`
     `INTEGER, SAVE :: cached_n = 0, cached_nblk = 0, cached_comm = 0`
   - If parameters match, reuse `cached_elpa_obj`.
   - In `finalize_elpa_library()`, properly deallocate `cached_elpa_obj` if associated.
3. **Fortran Line Length Strictness**:
   - Every line in `src/fm/cp_fm_elpa.F` must be $\le 132$ characters.
4. **Clean Git Commit**:
   - Commit cleanly on branch `feat/elpa-gpu-optimization`.
