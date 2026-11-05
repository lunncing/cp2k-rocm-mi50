# SPECIFICATION: GPU Offload of Grid Ops (copy, axpy), Hartree Decoupling, and PBE Functional

## Target Repository
Working tree: `D:/hermes/worktrees/cp2k-integrated`

## Objectives
Offload the 4 critical CPU bottlenecks to AMD GPU (HIP/ROCm) kernels:
1. `pw_copy`: Device-to-device fast copies and multi-grid `gidx` remapped copies on GPU.
2. `pw_axpy`: Device-to-device linear combination ($Y = \alpha X + \beta Y$) and `gidx` remapped axpy on GPU.
3. `modify_hartree_pot`: GPU offloaded phase-factor and Gaussian attenuation summation over atoms and G-vectors.
4. `pbe_lda_eval`: GPU offloaded point-wise PBE exchange-correlation energy and potential evaluation.

## Detailed Requirements

### 1. `src/pw/gpu/pw_gpu_kernels.cu` & `src/pw/gpu/pw_gpu_kernels.h`
- Implement HIP kernels:
  - `pw_gpu_copy_kernel`: contiguous DtoD copy or scale.
  - `pw_gpu_axpy_kernel`: vector $Y[i] = \alpha X[i] + \beta Y[i]$.
  - `pw_gpu_remap_copy_kernel`: $Y[i] = X[\text{gidx}[i]]$ or $Y[\text{gidx}[i]] = X[i]$.
  - `pw_gpu_remap_axpy_kernel`: $Y[i] = \alpha X[\text{gidx}[i]] + \beta Y[i]$.
  - `modify_hartree_pot_kernel`: parallel over G-vectors, summing atom phase factors $\exp(-i \vec{g} \cdot \vec{r}_a)$ and Gaussian decay.
  - `pbe_lda_eval_kernel`: point-wise PBE functional calculation on GPU over $N$ points.

### 2. `src/pw/gpu/pw_gpu_internal.c`
- Implement C wrappers:
  - `pw_gpu_copy_c(...)`
  - `pw_gpu_axpy_c(...)`
  - `pw_gpu_remap_copy_c(...)`
  - `pw_gpu_remap_axpy_c(...)`
  - `modify_hartree_pot_gpu_c(...)`
  - `pbe_lda_eval_gpu_c(...)`
- Maintain full compatibility with memory budget checks and host fallback.

### 3. Fortran Interfaces
- `src/pw/pw_gpu.F`:
  - Export Fortran wrappers with `BIND(C)`.
- Connect into:
  - `src/pw/pw_methods.F`: `pw_copy`, `pw_axpy`.
  - `src/cp_ddapc_util.F`: `modify_hartree_pot`.
  - `src/xc/xc_pbe.F`: `pbe_lda_calc`.

### 4. Constraints
- Fortran line lengths <= 132 chars.
- Exact bitwise numerical consistency or within standard machine epsilon ($10^{-12}$).
- Complete unit test suite in `tests/pw_gpu_integrated/`.
