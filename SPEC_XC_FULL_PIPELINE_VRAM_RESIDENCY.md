# SPECIFICATION: XC Full-Pipeline VRAM Residency (Gradient -> PBE -> Divergence Fusion)

## Target Repository
Working tree: `D:/hermes/worktrees/cp2k-integrated`

## Objective
Implement an end-to-end VRAM-resident XC execution pipeline that fuses:
1. Real-space Gradient computation ($\nabla\rho_{x,y,z}$ via inverse 3D-FFTs)
2. Gradient norm computation ($|\nabla\rho| = \sqrt{\nabla\rho_x^2 + \nabla\rho_y^2 + \nabla\rho_z^2}$)
3. PBE exchange-correlation functional energy density ($e_0$) and potential derivatives ($v_\rho$, $v_{\nabla\rho}$)
4. Divergence computation ($\nabla \cdot v_{\nabla\rho}$ via forward and inverse 3D-FFTs)
All intermediate arrays remain 100% resident in GPU VRAM (MI50 HBM2).
Eliminates 3.2 GB of intermediate PCIe transfers (DtoH and HtoD) and eliminates ~350 ms of CPU OpenMP PBE execution per SCF step.

## Detailed Architecture

### 1. `src/pw/gpu/pw_gpu_kernels.cu` & `src/pw/gpu/pw_gpu_kernels.h`
- Implement/verify HIP device kernels:
  - `pw_gpu_norm3_kernel`: computes `norm[i] = sqrt(g0[i]*g0[i] + g1[i]*g1[i] + g2[i]*g2[i])` for $i = 0 \dots N-1$.
  - `pw_gpu_pbe_lda_eval_kernel`: computes $e_0$, $v_\rho$, and $v_{\text{norm}}$ from `rho` and `norm`.
  - `pw_gpu_combine_div_input_kernel`: computes $v_{\nabla\rho, d}[i] = - v_{\text{norm}}[i] \cdot \frac{g_d[i]}{\max(\text{norm}[i], \epsilon)}$ in-place into `g_d[i]`.
  - `pw_gpu_add_vrho_kernel`: `vxc_out[i] += v_rho[i]`.

### 2. `src/pw/gpu/pw_gpu_internal.c`
- Pipeline function:
  ```c
  int pw_gpu_xc_full_pipeline_c(const double *zin_rho_g, const double *din_rho_r,
                                double *dout_vxc_r, double *exc_out,
                                const int *ghatmap, const double *g,
                                const int *npts, const int ngpts, const int nmaps,
                                const double scale, const int grid_id,
                                double epsilon_rho, double kappa, double beta, double mu,
                                double scale_ec, double scale_ex);
  ```
- Workflow inside `pw_gpu_xc_full_pipeline_c`:
  1. Check memory budget via `check_and_reclaim_memory` for:
     - 3x real scratch buffers `grad_dev[0..2]` ($3 \times \frac{1}{2}\text{buffer\_size} \approx 1.59\text{ GB}$)
     - 1x real scratch buffer `vrho_dev` ($\approx 531\text{ MB}$)
     - Standard work buffers `buffer_dev_1`, `buffer_dev_2`, `gbuf_dev`.
     If budget exceeded, return 0 (falls back to Fortran path).
  2. Upload `rho_g` to `gbuf_dev` (136 MB). Upload `rho_r` to `buffer_dev_1` (531 MB).
  3. Compute 3-axis gradient:
     For $d = 0, 1, 2$:
     - `offloadMemsetAsync(buffer_dev_2, 0, ...)`
     - `pw_gpu_launch_scatter_deriv(buffer_dev_2, gbuf_dev, g_dev, d, ...)`
     - `fft_3d(OFFLOAD_FFT_INVERSE, ..., buffer_dev_2)`
     - `pw_gpu_launch_complex_to_real(buffer_dev_2, grad_dev[d], nrpts, stream)`
  4. Compute norm directly in VRAM:
     - `pw_gpu_launch_norm3(grad_dev[0], grad_dev[1], grad_dev[2], norm_dev, nrpts, stream)`
  5. Compute PBE functional in VRAM:
     - `pw_gpu_launch_pbe_lda(buffer_dev_1, norm_dev, e0_dev, vrho_dev, vnorm_dev, nrpts, ...)`
     - Compute divergence input: `combine_div_input(grad_dev[0..2], vnorm_dev, norm_dev, nrpts)`
  6. Compute divergence in VRAM:
     - `offloadMemsetAsync(gbuf_dev, 0, gbuf_size, stream)`
     - For $d = 0, 1, 2$:
       - `pw_gpu_launch_real_to_complex(grad_dev[d], buffer_dev_2, nrpts, stream)`
       - `fft_3d(OFFLOAD_FFT_FORWARD, ..., buffer_dev_2)`
       - `pw_gpu_launch_gather_deriv_acc(gbuf_dev, buffer_dev_2, g_dev, d, scale, ...)`
     - Inverse FFT to real space:
       - `pw_gpu_launch_scatter(buffer_dev_2, gbuf_dev, 1.0, ...)`
       - `fft_3d(OFFLOAD_FFT_INVERSE, ..., buffer_dev_2)`
       - `pw_gpu_launch_complex_to_real(buffer_dev_2, dout_dev, nrpts, stream)`
     - Add $v_\rho$:
       - `pw_gpu_launch_add_vrho(dout_dev, vrho_dev, nrpts, stream)`
  7. Download only final `dout_vxc_r` (531 MB) and reduce `exc` to host.
  8. Stream synchronize and return 1.

### 3. `src/pw/pw_gpu.F` & `src/xc/xc.F`
- In `src/pw/pw_gpu.F`:
  - Export Fortran wrapper `pw_gpu_xc_full_pipeline(...)`.
- In `src/xc/xc.F`:
  - In `xc_vxc_pw_create`:
    For closed-shell PBE calculations without virial or laplace, check:
    ```fortran
    IF (pw_gpu_xc_full_pipeline_available() .AND. .NOT. lsd .AND. .NOT. use_virial) THEN
       success = pw_gpu_xc_full_pipeline(...)
       IF (success) RETURN
    END IF
    ```

### 4. Constraints
- Fortran line lengths <= 132 chars.
- 100% bitwise / numerical consistency within $10^{-12}$.
- Respect `MAX_GPU_MEMORY_MB` and `HOST_FALLBACK`.
- Add test in `tests/pw_gpu_integrated/`.
