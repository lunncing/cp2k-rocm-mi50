/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

#ifndef PW_GPU_KERNELS_H
#define PW_GPU_KERNELS_H

#include <stddef.h>

#define PW_GPU_XC_REDUCE_CHUNK 256
static inline size_t pw_gpu_xc_reduce_size(size_t n) {
  const size_t columns = n / PW_GPU_XC_REDUCE_CHUNK + (n % PW_GPU_XC_REDUCE_CHUNK != 0);
  return columns <= 1 ? columns : 2 * columns;
}

#include "../../offload/offload_runtime.h"
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
 * \brief Launcher for pw_real_to_complex kernel.
 * \author Ole Schuett
 ******************************************************************************/
void pw_gpu_launch_real_to_complex(const double *din, double *zout,
                                   const int ngpts, offloadStream_t stream);

/*******************************************************************************
 * \brief Launcher for pw_complex_to_real kernel.
 * \author Ole Schuett
 ******************************************************************************/
void pw_gpu_launch_complex_to_real(const double *zin, double *dout,
                                   const int ngpts, offloadStream_t stream);

/*******************************************************************************
 * \brief Launcher for pw_gather kernel.
 * \author Ole Schuett
 ******************************************************************************/
void pw_gpu_launch_gather(double *pwcc, const double *c, const double scale,
                          const int ngpts, const int *ghatmap,
                          offloadStream_t stream);

/*******************************************************************************
 * \brief Launcher for pw_scatter kernel.
 * \author Ole Schuett
 ******************************************************************************/
void pw_gpu_launch_scatter(double *c, const double *pwcc, const double scale,
                           const int ngpts, const int nmaps, const int *ghatmap,
                           offloadStream_t stream);

/*******************************************************************************
 * \brief Launcher for the fused derivative + scatter kernel.
 *        c[map(ig)] = scale * (0,1) * g(idir,ig) * zin(ig), conj. for map2.
 ******************************************************************************/
void pw_gpu_launch_scatter_deriv(double *c, const double *zin, const double *g,
                                 const int idir, const double scale,
                                 const int ngpts, const int nmaps,
                                 const int *ghatmap, offloadStream_t stream);

/*******************************************************************************
 * \brief Launcher for the fused gather + derivative + accumulation kernel.
 *        acc(ig) += (0,1) * g(idir,ig) * scale * c[map(ig)].
 ******************************************************************************/
void pw_gpu_launch_gather_deriv_acc(double *acc, const double *c,
                                    const double *g, const int idir,
                                    const double scale, const int ngpts,
                                    const int *ghatmap,
                                    offloadStream_t stream);

// Device-pointer launchers. cx/cy are the number of doubles per element (1/2).
void pw_gpu_launch_copy(const double *, double *, size_t, size_t, int, int,
                        double, offloadStream_t);
void pw_gpu_launch_axpy(const double *, double *, size_t, size_t, int, int,
                        double, double, offloadStream_t);
void pw_gpu_launch_remap_copy(const double *, double *, const int *, size_t,
                              int, int, int, double, offloadStream_t);
void pw_gpu_launch_remap_axpy(const double *, double *, const int *, size_t,
                              int, int, int, double, double, offloadStream_t);
void pw_gpu_launch_modify_hartree(double *, const double *, const double *,
    const double *, const double *, const double *, size_t, size_t, size_t,
    double, double, offloadStream_t);
void pw_gpu_launch_pbe_lda(const double *, const double *, double *, double *,
    double *, size_t, int, double, double, double, double, double, double,
    offloadStream_t);
void pw_gpu_launch_norm3(const double *, const double *, const double *,
                         double *, size_t, offloadStream_t);
void pw_gpu_launch_combine_div_input(double *, double *, double *,
    const double *, const double *, size_t, double, offloadStream_t);
void pw_gpu_launch_add_vrho(double *, const double *, size_t, offloadStream_t);
void pw_gpu_launch_xc_cutoff(double *, const double *, size_t, double, double,
                             offloadStream_t);
// Produces pw_gpu_xc_reduce_size(n) doubles: sum/correction pairs until the
// final scalar. Keeping the corrections avoids losing cancellation accuracy.
void pw_gpu_launch_xc_reduce(const double *, double *, size_t, offloadStream_t);

// Full VRAM pipeline. exc_out is the unscaled sum of energy density; the
// Fortran wrapper multiplies it by dvol. Outputs are untouched on failure.
// The basic entry uses epsilon_rho as the gradient cutoff, without filtering
// or density smoothing. The configured entry preserves xc_vxc_pw_create's
// independent gradient cutoff, spherical filter and density cutoff settings.
int pw_gpu_xc_full_pipeline_c(const double *zin_rho_g, const double *din_rho_r,
    double *dout_vxc_r, double *exc_out, const int *ghatmap, const double *g,
    const int *npts, int ngpts, int nmaps, double scale, int grid_id,
    double epsilon_rho, double kappa, double beta, double mu,
    double scale_ec, double scale_ex);
int pw_gpu_xc_full_pipeline_configured_c(const double *zin_rho_g,
    const double *din_rho_r, double *dout_vxc_r, double *exc_out,
    const int *ghatmap, const double *g, const int *npts, int ngpts, int nmaps,
    double scale, int grid_id, double epsilon_rho, double kappa, double beta,
    double mu, double scale_ec, double scale_ex, double gradient_cutoff,
    double smooth_range, int spherical);

// Synchronous host-pointer attempts: 1 = completed, 0 = leave all host outputs
// untouched and execute the original Fortran path. A memory failure observes
// HOST_FALLBACK (including fail-fast when disabled). Empty work is successful.
int pw_gpu_copy_c(const double *x, double *y, size_t nx, size_t ny, int cx,
                  int cy, double alpha);
int pw_gpu_axpy_c(const double *x, double *y, size_t nx, size_t ny, int cx,
                  int cy, double alpha, double beta);
int pw_gpu_remap_copy_c(const double *x, double *y, size_t nx, size_t ny, int cx,
                        int cy, const int *gidx, int scatter, double alpha);
int pw_gpu_remap_axpy_c(const double *x, double *y, size_t nx, size_t ny, int cx,
                        int cy, const int *gidx, int scatter, double alpha,
                        double beta);
// Hartree: g(3,n), r(3,natoms), cv(nradii,natoms). The caller selects the
// contiguous nonzero-G prefix up to GCUT and applies the separate G=0 term.
int modify_hartree_pot_gpu_c(double *v, const double *g, const double *gsq,
    const double *r, const double *radii, const double *cv, size_t n,
    size_t natoms, size_t nradii, double gcut2, double vol);
// Unpolarized PBE energy and dE/drho, dE/d|grad rho|, added to existing outputs.
// grad=0 does not access erho/enorm (the caller may alias them to rho).
int pbe_lda_eval_gpu_c(const double *rho, const double *norm, double *e0,
    double *erho, double *enorm, size_t n, int grad, double epsilon,
    double kappa, double beta, double mu, double sc, double sx);

#ifdef __cplusplus
}
#endif

#endif // defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
#endif

// EOF
