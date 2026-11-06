/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

#include "../../offload/offload_runtime.h"
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)

#include "pw_gpu_kernels.h"

#define PW_GPU_ELEMENTAL static __device__ inline
#include "pw_gpu_fused_ops.h"
#include "pw_gpu_pbe.h"

#if defined(_OMP_H)
#error "OpenMP should not be used in .cu files to accommodate HIP."
#endif

// Block size of the gather/scatter type kernels. These kernels are purely
// element-wise, so the block size has no influence on the results. It used to
// be 32, which occupies only half a wavefront on AMD GPUs (wave64).
#define PW_GPU_MAP_THREADS 256

/*******************************************************************************
 * \brief   Performs a out-of-place copy of a double precision vector (first
 *          half filled) into a double precision complex vector on the GPU.
 *          It requires a global double precision vector 'zout' of length '2n'.
 *          [memory (shared):  none Byte
 *           memory (private): 4 Byte
 *           memory (global):  16*n Byte]
 *          n - size of double precision input vector
 * \author  Andreas Gloess
 ******************************************************************************/
__global__ void pw_real_to_complex(const double *din, double *zout,
                                   const int ngpts) {
  const int igpt = blockIdx.x * blockDim.x + threadIdx.x;
  if (igpt < ngpts) {
    zout[2 * igpt] = din[igpt];
    zout[2 * igpt + 1] = 0.0;
  }
}

/*******************************************************************************
 * \brief Launcher for pw_real_to_complex kernel.
 * \author Ole Schuett
 ******************************************************************************/
void pw_gpu_launch_real_to_complex(const double *din, double *zout,
                                   const int ngpts, offloadStream_t stream) {
  const int threadsPerBlock = 1024;
  const int numBlocks = (ngpts + threadsPerBlock - 1) / threadsPerBlock;
  pw_real_to_complex<<<numBlocks, threadsPerBlock, 0, stream>>>(din, zout,
                                                                ngpts);
}

/*******************************************************************************
 * \brief   Performs a out-of-place copy of a double precision complex vector
 *          (real part) into a double precision vector on the GPU.
 *          It requires a global double precision vector 'dout' of length 'n'.
 *          [memory (shared):  none Byte
 *           memory (private): 4 Byte
 *           memory (global):  16*n Byte]
 *          n - size of double precision output vector
 * \author  Andreas Gloess
 ******************************************************************************/
__global__ void pw_complex_to_real(const double *zin, double *dout,
                                   const int ngpts) {
  const int igpt = blockIdx.x * blockDim.x + threadIdx.x;
  if (igpt < ngpts) {
    dout[igpt] = zin[2 * igpt];
  }
}

/*******************************************************************************
 * \brief Launcher for pw_complex_to_real kernel.
 * \author Ole Schuett
 ******************************************************************************/
void pw_gpu_launch_complex_to_real(const double *zin, double *dout,
                                   const int ngpts, offloadStream_t stream) {
  const int threadsPerBlock = 1024;
  const int numBlocks = (ngpts + threadsPerBlock - 1) / threadsPerBlock;
  pw_complex_to_real<<<numBlocks, threadsPerBlock, 0, stream>>>(zin, dout,
                                                                ngpts);
}

/*******************************************************************************
 * \brief   Performs a (double precision complex) gather and scale on the GPU.
 * \author  Andreas Gloess
 ******************************************************************************/
__global__ void pw_gather(double *pwcc, const double *c, const double scale,
                          const int ngpts, const int *ghatmap) {
  const int igpt = blockIdx.x * blockDim.x + threadIdx.x;
  if (igpt < ngpts) {
    pwcc[2 * igpt] = scale * c[2 * ghatmap[igpt]];
    pwcc[2 * igpt + 1] = scale * c[2 * ghatmap[igpt] + 1];
  }
}

/*******************************************************************************
 * \brief Launcher for pw_gather kernel.
 * \author Ole Schuett
 ******************************************************************************/
void pw_gpu_launch_gather(double *pwcc, const double *c, const double scale,
                          const int ngpts, const int *ghatmap,
                          offloadStream_t stream) {
  const int threadsPerBlock = PW_GPU_MAP_THREADS;
  const int numBlocks = (ngpts + threadsPerBlock - 1) / threadsPerBlock;
  pw_gather<<<numBlocks, threadsPerBlock, 0, stream>>>(pwcc, c, scale, ngpts,
                                                       ghatmap);
}

/*******************************************************************************
 * \brief   Performs a (double precision complex) scatter and scale on the GPU.
 * \author  Andreas Gloess
 ******************************************************************************/
__global__ void pw_scatter(double *c, const double *pwcc, const double scale,
                           const int ngpts, const int nmaps,
                           const int *ghatmap) {
  const int igpt = blockIdx.x * blockDim.x + threadIdx.x;
  if (igpt < ngpts) {
    c[2 * ghatmap[igpt]] = scale * pwcc[2 * igpt];
    c[2 * ghatmap[igpt] + 1] = scale * pwcc[2 * igpt + 1];
    if (nmaps == 2) {
      c[2 * ghatmap[igpt + ngpts]] = scale * pwcc[2 * igpt];
      c[2 * ghatmap[igpt + ngpts] + 1] = -scale * pwcc[2 * igpt + 1];
    }
  }
}

/*******************************************************************************
 * \brief Launcher for pw_scatter kernel.
 * \author Ole Schuett
 ******************************************************************************/
void pw_gpu_launch_scatter(double *c, const double *pwcc, const double scale,
                           const int ngpts, const int nmaps, const int *ghatmap,
                           offloadStream_t stream) {
  const int threadsPerBlock = PW_GPU_MAP_THREADS;
  const int numBlocks = (ngpts + threadsPerBlock - 1) / threadsPerBlock;
  pw_scatter<<<numBlocks, threadsPerBlock, 0, stream>>>(c, pwcc, scale, ngpts,
                                                        nmaps, ghatmap);
}

/*******************************************************************************
 * \brief   Fused derivative + scatter (see pw_gpu_fused_ops.h).
 ******************************************************************************/
__global__ void pw_scatter_deriv(double *c, const double *zin, const double *g,
                                 const int idir, const double scale,
                                 const int ngpts, const int nmaps,
                                 const int *ghatmap) {
  const int igpt = blockIdx.x * blockDim.x + threadIdx.x;
  if (igpt < ngpts) {
    pw_fused_scatter_deriv_elem(c, zin, g, idir, scale, igpt, ngpts, nmaps,
                                ghatmap);
  }
}

/*******************************************************************************
 * \brief Launcher for pw_scatter_deriv kernel.
 ******************************************************************************/
void pw_gpu_launch_scatter_deriv(double *c, const double *zin, const double *g,
                                 const int idir, const double scale,
                                 const int ngpts, const int nmaps,
                                 const int *ghatmap, offloadStream_t stream) {
  const int threadsPerBlock = PW_GPU_MAP_THREADS;
  const int numBlocks = (ngpts + threadsPerBlock - 1) / threadsPerBlock;
  pw_scatter_deriv<<<numBlocks, threadsPerBlock, 0, stream>>>(
      c, zin, g, idir, scale, ngpts, nmaps, ghatmap);
}

/*******************************************************************************
 * \brief   Fused gather + derivative + accumulation (see pw_gpu_fused_ops.h).
 ******************************************************************************/
__global__ void pw_gather_deriv_acc(double *acc, const double *c,
                                    const double *g, const int idir,
                                    const double scale, const int ngpts,
                                    const int *ghatmap) {
  const int igpt = blockIdx.x * blockDim.x + threadIdx.x;
  if (igpt < ngpts) {
    pw_fused_gather_deriv_acc_elem(acc, c, g, idir, scale, igpt, ghatmap);
  }
}

/*******************************************************************************
 * \brief Launcher for pw_gather_deriv_acc kernel.
 ******************************************************************************/
void pw_gpu_launch_gather_deriv_acc(double *acc, const double *c,
                                    const double *g, const int idir,
                                    const double scale, const int ngpts,
                                    const int *ghatmap,
                                    offloadStream_t stream) {
  const int threadsPerBlock = PW_GPU_MAP_THREADS;
  const int numBlocks = (ngpts + threadsPerBlock - 1) / threadsPerBlock;
  pw_gather_deriv_acc<<<numBlocks, threadsPerBlock, 0, stream>>>(
      acc, c, g, idir, scale, ngpts, ghatmap);
}

// Grid-stride loops also cover arrays whose double count exceeds INT_MAX.
__global__ void pw_gpu_copy_kernel(const double *x, double *y, size_t nx,
                                   size_t ny, int cx, int cy, double alpha) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < ny;
       i += (size_t)blockDim.x * gridDim.x) {
    for (int c = 0; c < cy; c++) {
      y[(size_t)cy * i + c] = i < nx
          ? pw_grid_scale(pw_grid_component(x, i, cx, c), alpha) : 0.0;
    }
  }
}

__global__ void pw_gpu_axpy_kernel(const double *x, double *y, size_t nx,
                                   size_t ny, int cx, int cy, double alpha,
                                   double beta) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < ny;
       i += (size_t)blockDim.x * gridDim.x) {
    for (int c = 0; c < cy; c++) {
      const size_t j = (size_t)cy * i + c;
      y[j] = pw_grid_axpy(i < nx && alpha != 0.0
                            ? pw_grid_component(x, i, cx, c) : 0.0,
                          beta != 0.0 ? y[j] : 0.0,
                          i < nx ? alpha : 0.0, beta);
    }
  }
}

__global__ void pw_gpu_remap_copy_kernel(const double *x, double *y,
    const int *gidx, size_t n, int cx, int cy, int scatter, double alpha) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)blockDim.x * gridDim.x) {
    const size_t src = scatter ? i : (size_t)gidx[i] - 1;
    const size_t dst = scatter ? (size_t)gidx[i] - 1 : i;
    for (int c = 0; c < cy; c++)
      y[(size_t)cy * dst + c] =
          pw_grid_scale(pw_grid_component(x, src, cx, c), alpha);
  }
}

__global__ void pw_gpu_remap_axpy_kernel(const double *x, double *y,
    const int *gidx, size_t n, int cx, int cy, int scatter, double alpha,
    double beta) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)blockDim.x * gridDim.x) {
    const size_t src = scatter ? i : (size_t)gidx[i] - 1;
    const size_t dst = scatter ? (size_t)gidx[i] - 1 : i;
    for (int c = 0; c < cy; c++) {
      const size_t j = (size_t)cy * dst + c;
      y[j] = pw_grid_axpy(alpha != 0.0 ? pw_grid_component(x, src, cx, c) : 0.0,
                          beta != 0.0 ? y[j] : 0.0, alpha, beta);
    }
  }
}

__global__ void modify_hartree_pot_kernel(double *v, const double *g,
    const double *gsq, const double *r, const double *radii, const double *cv,
    size_t n, size_t natoms, size_t nradii, double gcut2, double vol) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)blockDim.x * gridDim.x)
    pw_hartree_elem(v, g, gsq, r, radii, cv, i, natoms, nradii, gcut2, vol);
}

__global__ void pw_gpu_pbe_lda_eval_kernel(const double *rho, const double *norm,
    double *e0, double *erho, double *enorm, size_t n, int grad,
    double epsilon, double kappa, double beta, double mu, double sc, double sx) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)blockDim.x * gridDim.x)
    pw_pbe_lda_elem(rho, norm, e0, erho, enorm, i, grad, epsilon,
                    kappa, beta, mu, sc, sx);
}

static unsigned int pw_grid_blocks(size_t n) {
  const size_t blocks = n / PW_GPU_MAP_THREADS + (n % PW_GPU_MAP_THREADS != 0);
  return (unsigned int)(blocks < 65535 ? blocks : 65535);
}

void pw_gpu_launch_copy(const double *x, double *y, size_t nx, size_t ny,
    int cx, int cy, double alpha, offloadStream_t s) {
  if (ny) pw_gpu_copy_kernel<<<pw_grid_blocks(ny), PW_GPU_MAP_THREADS, 0, s>>>(
      x, y, nx, ny, cx, cy, alpha);
}

void pw_gpu_launch_axpy(const double *x, double *y, size_t nx, size_t ny,
    int cx, int cy, double alpha, double beta, offloadStream_t s) {
  if (ny) pw_gpu_axpy_kernel<<<pw_grid_blocks(ny), PW_GPU_MAP_THREADS, 0, s>>>(
      x, y, nx, ny, cx, cy, alpha, beta);
}

void pw_gpu_launch_remap_copy(const double *x, double *y, const int *map,
    size_t n, int cx, int cy, int scatter, double alpha, offloadStream_t s) {
  if (n) pw_gpu_remap_copy_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
      x, y, map, n, cx, cy, scatter, alpha);
}

void pw_gpu_launch_remap_axpy(const double *x, double *y, const int *map,
    size_t n, int cx, int cy, int scatter, double alpha, double beta,
    offloadStream_t s) {
  if (n) pw_gpu_remap_axpy_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
      x, y, map, n, cx, cy, scatter, alpha, beta);
}

void pw_gpu_launch_modify_hartree(double *v, const double *g, const double *gsq,
    const double *r, const double *radii, const double *cv, size_t n,
    size_t natoms, size_t nradii, double gcut2, double vol, offloadStream_t s) {
  if (n) modify_hartree_pot_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
      v, g, gsq, r, radii, cv, n, natoms, nradii, gcut2, vol);
}

void pw_gpu_launch_pbe_lda(const double *rho, const double *norm, double *e0,
    double *erho, double *enorm, size_t n, int grad, double epsilon,
    double kappa, double beta, double mu, double sc, double sx, offloadStream_t s) {
  if (n) pw_gpu_pbe_lda_eval_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
      rho, norm, e0, erho, enorm, n, grad, epsilon, kappa, beta, mu, sc, sx);
}

#include "pw_gpu_xc_ops.h"

__global__ void pw_gpu_norm3_kernel(const double *g0, const double *g1,
    const double *g2, double *norm, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)blockDim.x * gridDim.x)
    norm[i] = pw_xc_norm3(g0[i], g1[i], g2[i]);
}

__global__ void pw_gpu_combine_div_input_kernel(double *g0, double *g1,
    double *g2, const double *vnorm, const double *norm, size_t n, double cutoff) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)blockDim.x * gridDim.x) {
    const double factor = pw_xc_div_factor(vnorm[i], norm[i], cutoff);
    g0[i] = -PW_GPU_DMUL(g0[i], factor);
    g1[i] = -PW_GPU_DMUL(g1[i], factor);
    g2[i] = -PW_GPU_DMUL(g2[i], factor);
  }
}

__global__ void pw_gpu_add_vrho_kernel(double *vxc, const double *vrho, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)blockDim.x * gridDim.x)
    vxc[i] = PW_GPU_DADD(vrho[i], vxc[i]);
}

__global__ void pw_gpu_xc_cutoff_kernel(double *values, const double *rho,
    size_t n, double cutoff, double range) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)blockDim.x * gridDim.x)
    values[i] = pw_xc_cutoff(values[i], rho[i], cutoff, range);
}

__global__ void pw_gpu_xc_reduce_kernel(const double *values, double *partial,
                                        size_t n, size_t columns) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < columns;
       i += (size_t)blockDim.x * gridDim.x)
    pw_xc_sum_column(values, partial, n, i, columns);
}

void pw_gpu_launch_norm3(const double *g0, const double *g1, const double *g2,
                         double *norm, size_t n, offloadStream_t s) {
  if (n) pw_gpu_norm3_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
      g0, g1, g2, norm, n);
}

void pw_gpu_launch_combine_div_input(double *g0, double *g1, double *g2,
    const double *vnorm, const double *norm, size_t n, double cutoff,
    offloadStream_t s) {
  if (n) pw_gpu_combine_div_input_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
      g0, g1, g2, vnorm, norm, n, cutoff);
}

void pw_gpu_launch_add_vrho(double *vxc, const double *vrho, size_t n,
                            offloadStream_t s) {
  if (n) pw_gpu_add_vrho_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
      vxc, vrho, n);
}

void pw_gpu_launch_xc_cutoff(double *values, const double *rho, size_t n,
    double cutoff, double range, offloadStream_t s) {
  if (n && range > 0.0)
    pw_gpu_xc_cutoff_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
        values, rho, n, cutoff, range);
}

void pw_gpu_launch_xc_reduce(const double *values, double *partial, size_t n,
                             offloadStream_t s) {
  const size_t columns = n / PW_GPU_XC_REDUCE_CHUNK + (n % PW_GPU_XC_REDUCE_CHUNK != 0);
  if (n) pw_gpu_xc_reduce_kernel<<<pw_grid_blocks(columns), PW_GPU_MAP_THREADS, 0, s>>>(
      values, partial, n, columns);
}

#include "pw_gpu_res_ops.h"

__global__ void pw_gpu_res_cmul_kernel(double *out, const double *a,
                                       const double *w, int width, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)blockDim.x * gridDim.x)
    pw_res_cmul_elem(out, a, w, width, i);
}

__global__ void pw_gpu_res_cdot_terms_kernel(const double *a, const double *b,
                                             double *p, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)blockDim.x * gridDim.x)
    p[i] = pw_res_cdot_term(a, b, i);
}

__global__ void pw_gpu_res_sample_kernel(const double *x, double *out,
                                         size_t n, int width, size_t count) {
  for (size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x; k < count;
       k += (size_t)blockDim.x * gridDim.x) {
    const size_t e = pw_res_sample_index(k, n);
    for (int c = 0; c < width; c++)
      out[(size_t)width * k + c] = x[(size_t)width * e + c];
  }
}

void pw_gpu_launch_res_cmul(double *out, const double *a, const double *w,
                            int width, size_t n, offloadStream_t s) {
  if (n) pw_gpu_res_cmul_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
      out, a, w, width, n);
}

void pw_gpu_launch_res_cdot_terms(const double *a, const double *b, double *p,
                                  size_t n, offloadStream_t s) {
  if (n) pw_gpu_res_cdot_terms_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
      a, b, p, n);
}

void pw_gpu_launch_res_sample(const double *x, double *out, size_t n, int width,
                              offloadStream_t s) {
  const size_t count = pw_res_sample_count(n);
  if (count) pw_gpu_res_sample_kernel<<<pw_grid_blocks(count), PW_GPU_MAP_THREADS, 0, s>>>(
      x, out, n, width, count);
}

// Adapted from the verified natural_fft_kernels.cu prototype. Retain the
// separately rounded arithmetic used by the sorted-G Fortran operations.
__global__ void pw_gpu_res_natural_cmul_kernel(double *v_h, const double *rho,
    const double *infl, int width, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)blockDim.x * gridDim.x)
    pw_res_cmul_elem(v_h, rho, infl, width, i);
}

__global__ void pw_gpu_res_natural_cdot_terms_kernel(const double *rho,
    const double *v_h, double *terms, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)blockDim.x * gridDim.x)
    terms[i] = pw_res_cdot_term(rho, v_h, i);
}

__global__ void pw_gpu_res_natural_extract_prefix_kernel(const double *src,
    double *prefix_out, const int *coords, size_t nprefix) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < nprefix;
       i += (size_t)blockDim.x * gridDim.x) {
    const size_t j = (size_t)coords[i];
    prefix_out[2 * i] = src[2 * j];
    prefix_out[2 * i + 1] = src[2 * j + 1];
  }
}

void pw_gpu_launch_res_natural_cmul(double *v_h, const double *rho,
    const double *infl, int width, size_t n, offloadStream_t s) {
  if (n) pw_gpu_res_natural_cmul_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
      v_h, rho, infl, width, n);
}

void pw_gpu_launch_res_natural_cdot_terms(const double *rho, const double *v_h,
    double *terms, size_t n, offloadStream_t s) {
  if (n) pw_gpu_res_natural_cdot_terms_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
      rho, v_h, terms, n);
}

void pw_gpu_launch_res_natural_extract_prefix(const double *src,
    double *prefix_out, const int *coords, size_t nprefix, offloadStream_t s) {
  if (nprefix) pw_gpu_res_natural_extract_prefix_kernel<<<pw_grid_blocks(nprefix), PW_GPU_MAP_THREADS, 0, s>>>(
      src, prefix_out, coords, nprefix);
}

/*******************************************************************************
 * \brief Natural FFT order of XC (stage 4), see pw_gpu_res_ops.h. Maps are the
 *        ZERO-based g_hatmap.
 ******************************************************************************/
__global__ void pw_gpu_nat_deriv_kernel(double *c, const double *z,
    const double *gt, int mode, size_t n0, size_t n1, int idir, double scale,
    size_t n) {
  for (size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x; p < n;
       p += (size_t)blockDim.x * gridDim.x)
    pw_nat_deriv_elem(c, z, pw_nat_gd(gt, mode, n0, n1, p, idir), scale, p);
}

__global__ void pw_gpu_nat_deriv_acc_kernel(double *acc, const double *c,
    const double *gt, int mode, size_t n0, size_t n1, int idir, double scale,
    size_t n) {
  for (size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x; p < n;
       p += (size_t)blockDim.x * gridDim.x)
    pw_nat_deriv_acc_elem(acc, c, pw_nat_gd(gt, mode, n0, n1, p, idir), scale, p);
}

// Concurrent writes of one table entry store the same value if the g vectors
// are separable; pw_gpu_gtab_check_kernel decides whether they are.
__global__ void pw_gpu_gtab_build_kernel(double *gt, const double *g,
    const int *map, size_t n0, size_t n1, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)blockDim.x * gridDim.x) {
    const size_t p = (size_t)map[i], r = p / n0;
    gt[p % n0] = g[3 * i];
    gt[n0 + r % n1] = g[3 * i + 1];
    gt[n0 + n1 + r / n1] = g[3 * i + 2];
  }
}

__global__ void pw_gpu_gtab_check_kernel(double *flags, const double *g,
    const int *map, const double *gt, size_t n0, size_t n1, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)blockDim.x * gridDim.x) {
    const size_t p = (size_t)map[i];
    int bad = 0;
    for (int d = 0; d < 3; d++)
      bad |= pw_res_bits_differ(g[3 * i + d], pw_nat_gd(gt, 1, n0, n1, p, d));
    flags[i] = bad ? 1.0 : 0.0;
  }
}

__global__ void pw_gpu_gtab_full_kernel(double *gt, const double *g,
    const int *map, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)blockDim.x * gridDim.x) {
    const size_t p = (size_t)map[i];
    gt[3 * p] = g[3 * i];
    gt[3 * p + 1] = g[3 * i + 1];
    gt[3 * p + 2] = g[3 * i + 2];
  }
}

void pw_gpu_launch_nat_deriv(double *c, const double *z, const double *gt,
    int mode, size_t n0, size_t n1, int idir, double scale, size_t n,
    offloadStream_t s) {
  if (n) pw_gpu_nat_deriv_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
      c, z, gt, mode, n0, n1, idir, scale, n);
}

void pw_gpu_launch_nat_deriv_acc(double *acc, const double *c, const double *gt,
    int mode, size_t n0, size_t n1, int idir, double scale, size_t n,
    offloadStream_t s) {
  if (n) pw_gpu_nat_deriv_acc_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
      acc, c, gt, mode, n0, n1, idir, scale, n);
}

void pw_gpu_launch_gtab_build(double *gt, const double *g, const int *map,
    size_t n0, size_t n1, size_t n, offloadStream_t s) {
  if (n) pw_gpu_gtab_build_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
      gt, g, map, n0, n1, n);
}

void pw_gpu_launch_gtab_check(double *flags, const double *g, const int *map,
    const double *gt, size_t n0, size_t n1, size_t n, offloadStream_t s) {
  if (n) pw_gpu_gtab_check_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
      flags, g, map, gt, n0, n1, n);
}

void pw_gpu_launch_gtab_full(double *gt, const double *g, const int *map,
    size_t n, offloadStream_t s) {
  if (n) pw_gpu_gtab_full_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
      gt, g, map, n);
}

/*******************************************************************************
 * \brief Stage 5: two real fields per FFT in the natural XC, see
 *        pw_gpu_res_ops.h, and the fused Hartree begin.
 ******************************************************************************/
__global__ void pw_gpu_nat_grad2_kernel(double *z, const double *rho, const double *gt,
    int mode, size_t n0, size_t n1, size_t n2, int a, int b, size_t n) {
  for (size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x; p < n;
       p += (size_t)blockDim.x * gridDim.x)
    pw_nat_grad2_elem(z, rho, gt, mode, n0, n1, n2, a, b, p);
}

__global__ void pw_gpu_nat_div2_kernel(double *acc, const double *w, const double *gt,
    int mode, size_t n0, size_t n1, size_t n2, int a, int b, double scale,
    int accumulate, size_t n) {
  for (size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x; p < n;
       p += (size_t)blockDim.x * gridDim.x)
    pw_nat_div2_elem(acc, w, gt, mode, n0, n1, n2, a, b, scale, accumulate, p);
}

__global__ void pw_gpu_pack2_kernel(const double *x, const double *y, double *c,
    size_t n) {
  for (size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x; p < n;
       p += (size_t)blockDim.x * gridDim.x) {
    c[2 * p] = x[p];
    c[2 * p + 1] = y[p];
  }
}

__global__ void pw_gpu_unpack2_kernel(const double *c, double *x, double *y,
    size_t n) {
  for (size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x; p < n;
       p += (size_t)blockDim.x * gridDim.x) {
    x[p] = c[2 * p];
    y[p] = c[2 * p + 1];
  }
}

__global__ void pw_gpu_res_tot_cmul_kernel(double *rho_tot, double *out,
    const double *core, const double *rg, const double *w, int width, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)blockDim.x * gridDim.x)
    pw_res_tot_cmul_elem(rho_tot, out, core, rg, w, width, i);
}

__global__ void pw_gpu_nat_rho_dz_kernel(double *z, const double *rho, const double *gt,
    int mode, size_t n0, size_t n1, size_t n2, size_t n) {
  for (size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x; p < n;
       p += (size_t)blockDim.x * gridDim.x)
    pw_nat_rho_dz_elem(z, rho, gt, mode, n0, n1, n2, p);
}

void pw_gpu_launch_nat_rho_dz(double *z, const double *rho, const double *gt, int mode,
    size_t n0, size_t n1, size_t n2, size_t n, offloadStream_t s) {
  if (n) pw_gpu_nat_rho_dz_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
      z, rho, gt, mode, n0, n1, n2, n);
}

void pw_gpu_launch_nat_grad2(double *z, const double *rho, const double *gt, int mode,
    size_t n0, size_t n1, size_t n2, int a, int b, size_t n, offloadStream_t s) {
  if (n) pw_gpu_nat_grad2_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
      z, rho, gt, mode, n0, n1, n2, a, b, n);
}

void pw_gpu_launch_nat_div2(double *acc, const double *w, const double *gt, int mode,
    size_t n0, size_t n1, size_t n2, int a, int b, double scale, int accumulate,
    size_t n, offloadStream_t s) {
  if (n) pw_gpu_nat_div2_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
      acc, w, gt, mode, n0, n1, n2, a, b, scale, accumulate, n);
}

void pw_gpu_launch_pack2(const double *x, const double *y, double *c, size_t n,
    offloadStream_t s) {
  if (n) pw_gpu_pack2_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(x, y, c, n);
}

void pw_gpu_launch_unpack2(const double *c, double *x, double *y, size_t n,
    offloadStream_t s) {
  if (n) pw_gpu_unpack2_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(c, x, y, n);
}

void pw_gpu_launch_res_tot_cmul(double *rho_tot, double *out, const double *core,
    const double *rg, const double *w, int width, size_t n, offloadStream_t s) {
  if (n) pw_gpu_res_tot_cmul_kernel<<<pw_grid_blocks(n), PW_GPU_MAP_THREADS, 0, s>>>(
      rho_tot, out, core, rg, w, width, n);
}

#endif // defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)

// EOF
