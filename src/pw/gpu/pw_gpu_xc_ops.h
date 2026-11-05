/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef PW_GPU_XC_OPS_H
#define PW_GPU_XC_OPS_H

#include "pw_gpu_grid_ops.h"

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
