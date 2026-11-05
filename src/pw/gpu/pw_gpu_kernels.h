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

// Device-resident PW chain (see pw_gpu_res_ops.h). out may alias a.
void pw_gpu_launch_res_cmul(double *out, const double *a, const double *w,
                            int width, size_t n, offloadStream_t stream);
// p[i] = Re(conj(a_i) b_i), reduced afterwards with pw_gpu_launch_xc_reduce.
void pw_gpu_launch_res_cdot_terms(const double *a, const double *b, double *p,
                                  size_t n, offloadStream_t stream);
// Gathers the pw_res_sample_count(n) fingerprint elements of x (width doubles
// per element) into out.
void pw_gpu_launch_res_sample(const double *x, double *out, size_t n, int width,
                              offloadStream_t stream);

// Natural FFT box order. Influence width is 1 (real) or 2 (complex).
// Prefix coordinates are validated, zero-based FFT box offsets.
void pw_gpu_launch_res_natural_cmul(double *v_h, const double *rho,
    const double *infl, int width, size_t n, offloadStream_t stream);
void pw_gpu_launch_res_natural_cdot_terms(const double *rho, const double *v_h,
    double *terms, size_t n, offloadStream_t stream);
void pw_gpu_launch_res_natural_extract_prefix(const double *src,
    double *prefix_out, const int *coords, size_t nprefix, offloadStream_t stream);

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
// Device residency of the PW chain (pw_gpu_internal.c; Fortran in pw_gpu.F).
// 1 = done, 0 = not done with all host outputs untouched (Hartree steps: the
// chain is abandoned and the caller recomputes it on the original path).
int pw_gpu_res_enabled_c(int what);
void pw_gpu_res_invalidate_c(void);
// Slot IDs: 0 RHO_G, 1 RHO_R, 2 VXC. -1 synchronizes any owner of host_ptr.
void pw_gpu_res_ensure_host_c(int id, void *host_ptr);
int pw_gpu_res_matches_c(int id, const void *host, size_t n);
int pw_gpu_res_integrate_r_c(double *total_rho_out, double dvol);
int pw_gpu_res_density_c(int nlevels, void *const *r_host, const int *npts,
    const int *ngpts, const int *nmaps, const int *grid_id, void *const *maps,
    void *const *gidx, const double *scale, const int *t_npts, int t_ngpts,
    int t_nmaps, int t_grid_id, const int *t_map, double *rho_g_host,
    double *rho_r_host);
int pw_gpu_res_potential_c(const double *v_host, int nlevels, void *const *r_host,
    const int *npts, const int *ngpts, const int *nmaps, const int *grid_id,
    void *const *maps, void *const *gidx, const double *rscale, int aux,
    double fwd_scale);
int pw_gpu_res_vtot_prepare_c(const double *vh_host, const double *vxc_host,
    size_t n, double dvol);
int pw_gpu_res_vtot_commit_c(const double *vtot_host, size_t n);
int pw_gpu_res_vtot_skip_host_c(void);
// Lazy window of the SCF driver (on != 0 opens it; closing writes every live
// result to its host array), the one deferred density of an SCF iteration, and
// release of a host array that is not read again (drops its device copy).
void pw_gpu_res_set_lazy_c(int on);
void pw_gpu_res_defer_density_c(int on);
void pw_gpu_res_discard_c(const void *host_ptr);
int pw_gpu_res_hartree_begin_c(const double *core_host, const double *rho_g_host,
    const double *infl_host, double *rho_tot_host, size_t ngpts, size_t nprefix,
    const int *npts);
int pw_gpu_res_hartree_modify_c(double g0_delta, int have_g0, const double *g,
    const double *gsq, const double *r, const double *radii, const double *cv,
    size_t first, size_t n, size_t natoms, size_t nradii, double gcut2, double vol);
int pw_gpu_res_hartree_energy_c(double *dot);
int pw_gpu_res_hartree_finish_c(double *v_h_r_host, const int *npts, int ngpts,
    int nmaps, int grid_id, const int *ghatmap, double dvol);
void pw_gpu_res_hartree_cancel_c(void);

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
