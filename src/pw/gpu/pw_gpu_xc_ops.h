/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef PW_GPU_XC_OPS_H
#define PW_GPU_XC_OPS_H

#include "pw_gpu_grid_ops.h"
#include "pw_gpu_pbe.h"

PW_GPU_ELEMENTAL double pw_xc_norm3(double x, double y, double z) {
  PW_GPU_NO_CONTRACT
  return sqrt(PW_GPU_DADD(PW_GPU_DADD(PW_GPU_DMUL(x, x), PW_GPU_DMUL(y, y)),
                          PW_GPU_DMUL(z, z)));
}

PW_GPU_ELEMENTAL double pw_xc_div_factor(double vnorm, double norm,
                                         double cutoff) {
  // divide_by_norm_drho, multiplication by the gradient, then pw_scale(-1).
  const double denominator = fmax(norm, cutoff);
  return denominator == 0.0 ? 0.0 : vnorm / denominator;
}

// The closed-shell, no-e_0 branch of xc:smooth_cutoff, including its exact
// multiplication order. Called for energy before reduction and Vxc after filter.
PW_GPU_ELEMENTAL double pw_xc_cutoff(double value, double rho,
                                     double cutoff, double range) {
  PW_GPU_NO_CONTRACT
  const double upper = PW_GPU_DMUL(cutoff, range);
  const double middle = PW_GPU_DADD(cutoff, upper) / 2.0;
  const double width = middle - cutoff;
  if (range > 0.0 && rho < upper) {
    if (rho < cutoff) return 0.0;
    if (rho < middle) {
      const double x = (rho - cutoff) / width;
      const double x2 = PW_GPU_DMUL(x, x);
      return PW_GPU_DMUL(PW_GPU_DMUL(value, x2), x - PW_GPU_DMUL(0.5, x2));
    }
    const double x = 2.0 - (rho - cutoff) / width;
    const double x2 = PW_GPU_DMUL(x, x);
    return PW_GPU_DMUL(value, 1.0 - PW_GPU_DMUL(x2, x - PW_GPU_DMUL(0.5, x2)));
  }
  return value;
}

PW_GPU_ELEMENTAL double pw_xc_cutoff_lsd(double value, double ra, double rb,
                                         double cutoff, double range) {
  return pw_xc_cutoff(value, PW_GPU_DADD(ra, rb), cutoff, range);
}

PW_GPU_ELEMENTAL void pw_xc_pbe_lsd_combine_elem(
    const double *ra, const double *rb, double *ga0, double *ga1, double *ga2,
    double *gb0, double *gb1, double *gb2, double *va, double *vb,
    double *energy, size_t point, double epsilon, double sx, double sc,
    double cutoff, double smooth) {
  PW_GPU_NO_CONTRACT
  const double norm_a = pw_xc_norm3(ga0[point], ga1[point], ga2[point]);
  const double norm_b = pw_xc_norm3(gb0[point], gb1[point], gb2[point]);
  const double dr0 = PW_GPU_DADD(ga0[point], gb0[point]);
  const double dr1 = PW_GPU_DADD(ga1[point], gb1[point]);
  const double dr2 = PW_GPU_DADD(ga2[point], gb2[point]);
  const double norm = pw_xc_norm3(dr0, dr1, dr2);
  double e0 = 0.0, era = 0.0, erb = 0.0, en = 0.0, ena = 0.0, enb = 0.0;
  pw_pbe_lsd_elem(ra[point], rb[point], norm, norm_a, norm_b, epsilon, sx, sc,
                  1, &e0, &era, &erb, &en, &ena, &enb);
  const double fa = pw_xc_div_factor(ena, norm_a, cutoff);
  const double fb = pw_xc_div_factor(enb, norm_b, cutoff);
  const double ftot = pw_xc_div_factor(en, norm, cutoff);
  const double ptr0 = PW_GPU_DMUL(dr0, ftot);
  const double ptr1 = PW_GPU_DMUL(dr1, ftot);
  const double ptr2 = PW_GPU_DMUL(dr2, ftot);
  ga0[point] = -PW_GPU_DADD(PW_GPU_DMUL(fa, ga0[point]), ptr0);
  ga1[point] = -PW_GPU_DADD(PW_GPU_DMUL(fa, ga1[point]), ptr1);
  ga2[point] = -PW_GPU_DADD(PW_GPU_DMUL(fa, ga2[point]), ptr2);
  gb0[point] = -PW_GPU_DADD(PW_GPU_DMUL(fb, gb0[point]), ptr0);
  gb1[point] = -PW_GPU_DADD(PW_GPU_DMUL(fb, gb1[point]), ptr1);
  gb2[point] = -PW_GPU_DADD(PW_GPU_DMUL(fb, gb2[point]), ptr2);
  energy[point] = pw_xc_cutoff_lsd(e0, ra[point], rb[point], epsilon, smooth);
  va[point] = era;
  vb[point] = erb;
}

// Each thread sums one strided column. Loads are coalesced, no atomics or
// cross-block synchronization are needed, and the summation order is fixed.
PW_GPU_ELEMENTAL void pw_xc_sum_column(const double *x, double *partial,
                                       size_t n, size_t column, size_t columns) {
  PW_GPU_NO_CONTRACT
  double sum = 0.0, correction = 0.0;
  for (size_t j = column; j < n; j += columns) {
    const double value = x[j];
    const double next = PW_GPU_DADD(sum, value);
    const double error = fabs(sum) >= fabs(value)
        ? PW_GPU_DADD(PW_GPU_DADD(sum, -next), value)
        : PW_GPU_DADD(PW_GPU_DADD(value, -next), sum);
    correction = PW_GPU_DADD(correction, error);
    sum = next;
  }
  if (columns == 1) partial[0] = PW_GPU_DADD(sum, correction);
  else {
    partial[column] = sum;
    partial[column + columns] = correction;
  }
}

#endif
