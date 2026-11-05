/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

/*******************************************************************************
 * \brief Element operations of the fused GPU kernels (XC gradient/divergence).
 *
 *        The includer must define PW_GPU_ELEMENTAL, e.g.
 *          pw_gpu_kernels.cu : #define PW_GPU_ELEMENTAL static __device__ inline
 *          host unit test    : #define PW_GPU_ELEMENTAL static inline
 *
 *        Bitwise reproducibility with the unfused code path:
 *          pw_gather  (GPU) : t   = scale * c(map)
 *          pw_derive  (CPU) : t   = t * g(idir) ;  t = (0,1) * t = (-t_im, t_re)
 *          pw_axpy    (CPU) : acc = acc + t
 *          pw_scatter (GPU) : c(map1) = scale * t ; c(map2) = conj(scale * t)
 *        Each of these is a single rounded multiply or add. Contracting them
 *        into fused multiply-adds would change the last bit, therefore FP
 *        contraction is switched off inside the element functions.
 ******************************************************************************/
#ifndef PW_GPU_FUSED_OPS_H
#define PW_GPU_FUSED_OPS_H

#include <stddef.h>

#if !defined(PW_GPU_ELEMENTAL)
#error "PW_GPU_ELEMENTAL has to be defined before including pw_gpu_fused_ops.h"
#endif

#if defined(__clang__)
#define PW_GPU_NO_CONTRACT _Pragma("clang fp contract(off)")
#else
#define PW_GPU_NO_CONTRACT
#endif

#if defined(__CUDACC__) && !defined(__HIP__) && defined(__CUDA_ARCH__)
// nvcc device code: use the explicitly rounded intrinsics (never fused).
#define PW_GPU_DMUL(a, b) __dmul_rn((a), (b))
#define PW_GPU_DADD(a, b) __dadd_rn((a), (b))
#else
#define PW_GPU_DMUL(a, b) ((a) * (b))
#define PW_GPU_DADD(a, b) ((a) + (b))
#endif

/*******************************************************************************
 * \brief One G-vector of: scatter( scale * (0,1) * g(idir) * zin ).
 *        Replaces pw_copy + pw_derive (CPU) + upload + pw_scatter (GPU).
 *        c must be zeroed beforehand (like in the unfused scatter path).
 ******************************************************************************/
PW_GPU_ELEMENTAL void pw_fused_scatter_deriv_elem(
    double *c, const double *zin, const double *g, const int idir,
    const double scale, const int igpt, const int ngpts, const int nmaps,
    const int *ghatmap) {
  PW_GPU_NO_CONTRACT
  const size_t ig = (size_t)igpt;
  const double gd = g[3 * ig + (size_t)idir];
  // pw_derive: multiply by the real g(idir) ...
  const double tr = PW_GPU_DMUL(zin[2 * ig], gd);
  const double ti = PW_GPU_DMUL(zin[2 * ig + 1], gd);
  // ... and by the imaginary unit: (tr, ti) * (0, 1) = (-ti, tr).
  const double dr = -ti;
  const double di = tr;
  // pw_scatter
  const size_t i1 = (size_t)ghatmap[ig];
  c[2 * i1] = PW_GPU_DMUL(scale, dr);
  c[2 * i1 + 1] = PW_GPU_DMUL(scale, di);
  if (nmaps == 2) {
    const size_t i2 = (size_t)ghatmap[ig + (size_t)ngpts];
    c[2 * i2] = PW_GPU_DMUL(scale, dr);
    c[2 * i2 + 1] = PW_GPU_DMUL(-scale, di);
  }
}

/*******************************************************************************
 * \brief One G-vector of: acc += (0,1) * g(idir) * gather(scale * c).
 *        Replaces download + pw_derive + pw_axpy (CPU) of xc_pw_divergence.
 *        acc must be zeroed before the first direction (like pw_zero(vxc_g)).
 ******************************************************************************/
PW_GPU_ELEMENTAL void
pw_fused_gather_deriv_acc_elem(double *acc, const double *c, const double *g,
                               const int idir, const double scale,
                               const int igpt, const int *ghatmap) {
  PW_GPU_NO_CONTRACT
  const size_t ig = (size_t)igpt;
  const size_t i1 = (size_t)ghatmap[ig];
  // pw_gather
  const double pr = PW_GPU_DMUL(scale, c[2 * i1]);
  const double pi = PW_GPU_DMUL(scale, c[2 * i1 + 1]);
  // pw_derive
  const double gd = g[3 * ig + (size_t)idir];
  const double tr = PW_GPU_DMUL(pr, gd);
  const double ti = PW_GPU_DMUL(pi, gd);
  // pw_axpy with alpha = 1: acc = acc + (-ti, tr)
  acc[2 * ig] = PW_GPU_DADD(acc[2 * ig], -ti);
  acc[2 * ig + 1] = PW_GPU_DADD(acc[2 * ig + 1], tr);
}

#endif

// EOF
