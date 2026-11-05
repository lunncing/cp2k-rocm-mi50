/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef PW_GPU_GRID_OPS_H
#define PW_GPU_GRID_OPS_H

/* Element operations shared with the host kernel harness. The caller defines
 * PW_GPU_ELEMENTAL. Keep every multiply/add rounded separately, as in Fortran.
 * Complex values are interleaved doubles; gidx is Fortran's ONE-based map.
 * Scatter maps must be injective (the same requirement as the OMP CPU loops).
 */
#include "pw_gpu_fused_ops.h"
#include <math.h>

PW_GPU_ELEMENTAL double pw_grid_component(const double *x, size_t i, int cx,
                                          int component) {
  return component < cx ? x[(size_t)cx * i + component] : 0.0;
}

PW_GPU_ELEMENTAL double pw_grid_scale(double x, double alpha) {
  return alpha == 1.0 ? x : PW_GPU_DMUL(alpha, x);
}

PW_GPU_ELEMENTAL double pw_grid_axpy(double x, double y, double alpha,
                                    double beta) {
  PW_GPU_NO_CONTRACT
  // pw_axpy first scales/zeros Y, then adds X. In particular beta=0 must
  // overwrite NaNs, and alpha=0 must not propagate NaNs from X.
  y = beta == 0.0 ? 0.0 : pw_grid_scale(y, beta);
  return alpha == 0.0 ? y : PW_GPU_DADD(y, pw_grid_scale(x, alpha));
}

PW_GPU_ELEMENTAL void pw_hartree_elem(
    double *v, const double *g, const double *gsq, const double *r,
    const double *radii, const double *cv, size_t ig, size_t natoms,
    size_t nradii, double gcut2, double vol) {
  PW_GPU_NO_CONTRACT
  const double g2 = gsq[ig];
  const double delta = g2 - gcut2;
  const double pi = 3.141592653589793238462643383279502884;
  const double w = PW_GPU_DMUL(PW_GPU_DMUL(4.0, pi),
                              PW_GPU_DMUL(delta, delta)) /
                   PW_GPU_DMUL(g2, gcut2);
  double re = 0.0, im = 0.0;
  // Preserve the atom-major, Gaussian-minor summation order. No reduction.
  for (size_t a = 0; a < natoms; a++) {
    const double arg = PW_GPU_DADD(
        PW_GPU_DADD(PW_GPU_DMUL(g[3 * ig], r[3 * a]),
                    PW_GPU_DMUL(g[3 * ig + 1], r[3 * a + 1])),
        PW_GPU_DMUL(g[3 * ig + 2], r[3 * a + 2]));
    const double pr = cos(arg), pi_part = -sin(arg);
    for (size_t j = 0; j < nradii; j++) {
      const double rc2 = PW_GPU_DMUL(radii[j], radii[j]);
      const double gaussian = exp(PW_GPU_DMUL(-g2, rc2) / 4.0);
      const double c = PW_GPU_DMUL(gaussian, cv[a * nradii + j]);
      re = PW_GPU_DADD(re, PW_GPU_DMUL(c, pr));
      im = PW_GPU_DADD(im, PW_GPU_DMUL(c, pi_part));
    }
  }
  const double sfac = -1.0 / vol;
  v[2 * ig] = PW_GPU_DADD(v[2 * ig],
                          PW_GPU_DMUL(sfac, PW_GPU_DMUL(re, w)) / vol);
  v[2 * ig + 1] = PW_GPU_DADD(v[2 * ig + 1],
                              PW_GPU_DMUL(sfac, PW_GPU_DMUL(im, w)) / vol);
}

#endif
