/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations          */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>                 */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                   */
/*----------------------------------------------------------------------------*/
#ifndef PW_GPU_HOST_H
#define PW_GPU_HOST_H

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PW_GPU_ELEMENTAL static inline
#include "pw_gpu_fused_ops.h"
#undef PW_GPU_ELEMENTAL

// A dependency-free host path is needed even in builds without FFTW. These
// mixed-radix, unnormalized FFTs use the same signs and layouts as the device
// plans. Smooth CP2K grid dimensions take O(N log N) work; a prime radix uses
// a direct DFT. This path prioritizes bounded device memory over performance.
static double *host_buffer(const size_t doubles) {
  double *p = calloc(doubles, sizeof(double));
  if (p == NULL) {
    fprintf(stderr, "[PW_GPU] Unable to allocate host FFT scratch.\n");
    abort();
  }
  return p;
}

static void host_fft_recursive(const double *in, const size_t stride,
                               double *out, double *scratch, const int n,
                               const int sign) {
  if (n == 1) {
    out[0] = in[0];
    out[1] = in[1];
    return;
  }
  int radix = 2;
  while (radix <= n / radix && n % radix != 0) {
    radix++;
  }
  if (n % radix != 0) {
    radix = n;
  }
  const int m = n / radix;
  for (int q = 0; q < radix; q++) {
    host_fft_recursive(in + 2 * (size_t)q * stride, stride * radix,
                       out + 2 * (size_t)q * m, scratch, m, sign);
  }
  const double angle = sign * 6.283185307179586476925286766559 / n;
  for (int k = 0; k < n; k++) {
    double re = 0.0, im = 0.0;
    for (int q = 0; q < radix; q++) {
      const double c = cos(angle * q * k), s = sin(angle * q * k);
      const size_t j = 2 * ((size_t)q * m + k % m);
      re += out[j] * c - out[j + 1] * s;
      im += out[j] * s + out[j + 1] * c;
    }
    scratch[2 * (size_t)k] = re;
    scratch[2 * (size_t)k + 1] = im;
  }
  memcpy(out, scratch, 2 * sizeof(double) * (size_t)n);
}

// Forward: interleaved input rays -> contiguous output rays; inverse reverses
// that transpose. Keeping a temporary also permits aliased public arguments.
static void host_fft_1d(const int sign, const int n, const int m,
                        const double *in, double *out) {
  double *result = host_buffer(2 * (size_t)n * m);
  double *scratch = host_buffer(2 * (size_t)n);
  for (int ray = 0; ray < m; ray++) {
    const size_t offset = sign < 0 ? (size_t)ray : (size_t)ray * n;
    host_fft_recursive(in + 2 * offset, sign < 0 ? (size_t)m : 1,
                       result + 2 * (size_t)ray * n, scratch, n, sign);
  }
  if (sign < 0) {
    memcpy(out, result, 2 * sizeof(double) * (size_t)n * m);
  } else {
    for (int ray = 0; ray < m; ray++) {
      for (int i = 0; i < n; i++) {
        const size_t src = 2 * ((size_t)ray * n + i);
        const size_t dst = 2 * ((size_t)i * m + ray);
        out[dst] = result[src];
        out[dst + 1] = result[src + 1];
      }
    }
  }
  free(scratch);
  free(result);
}

// npts uses Fortran order: x is contiguous, exactly as fft_3d(nz,ny,nx).
static void host_fft_3d(double *data, const int *npts, const int sign) {
  const size_t total = (size_t)npts[0] * npts[1] * npts[2];
  size_t stride = 1;
  for (int axis = 0; axis < 3; axis++) {
    const int n = npts[axis];
    double *out = host_buffer(2 * (size_t)n);
    double *scratch = host_buffer(2 * (size_t)n);
    const size_t block = stride * n;
    for (size_t base = 0; base < total; base += block) {
      for (size_t ray = 0; ray < stride; ray++) {
        host_fft_recursive(data + 2 * (base + ray), stride, out, scratch, n, sign);
        for (int i = 0; i < n; i++) {
          const size_t dst = 2 * (base + ray + (size_t)i * stride);
          data[dst] = out[2 * (size_t)i];
          data[dst + 1] = out[2 * (size_t)i + 1];
        }
      }
    }
    free(scratch);
    free(out);
    stride = block;
  }
}

static void host_gather(double *out, const double *in, const int *map,
                        const int ngpts, const double scale) {
  for (int i = 0; i < ngpts; i++) {
    out[2 * (size_t)i] = scale * in[2 * (size_t)map[i]];
    out[2 * (size_t)i + 1] = scale * in[2 * (size_t)map[i] + 1];
  }
}

static void host_scatter(double *out, const double *in, const int *map,
                         const int ngpts, const int nmaps, const double scale) {
  for (int i = 0; i < ngpts; i++) {
    out[2 * (size_t)map[i]] = scale * in[2 * (size_t)i];
    out[2 * (size_t)map[i] + 1] = scale * in[2 * (size_t)i + 1];
    if (nmaps == 2) {
      out[2 * (size_t)map[i + ngpts]] = scale * in[2 * (size_t)i];
      out[2 * (size_t)map[i + ngpts] + 1] = -scale * in[2 * (size_t)i + 1];
    }
  }
}

static void host_cfffg(const double *in, double *out, const int *map,
                       const int *npts, const int ngpts, const double scale) {
  const size_t n = (size_t)npts[0] * npts[1] * npts[2];
  double *work = host_buffer(2 * n);
  for (size_t i = 0; i < n; i++) {
    work[2 * i] = in[i];
  }
  host_fft_3d(work, npts, -1);
  host_gather(out, work, map, ngpts, scale);
  free(work);
}

static void host_sfffc(const double *in, double *out, const int *map,
                       const int *npts, const int ngpts, const int nmaps,
                       const double scale) {
  const size_t n = (size_t)npts[0] * npts[1] * npts[2];
  double *work = host_buffer(2 * n);
  host_scatter(work, in, map, ngpts, nmaps, scale);
  host_fft_3d(work, npts, 1);
  for (size_t i = 0; i < n; i++) {
    out[i] = work[2 * i];
  }
  free(work);
}

static void host_grad3(const double *in, double *out[3], const int *map,
                       const double *g, const int *npts, const int ngpts,
                       const int nmaps) {
  const size_t n = (size_t)npts[0] * npts[1] * npts[2];
  double *work = host_buffer(2 * n);
  for (int d = 0; d < 3; d++) {
    memset(work, 0, 2 * sizeof(double) * n);
    for (int i = 0; i < ngpts; i++) {
      pw_fused_scatter_deriv_elem(work, in, g, d, 1.0, i, ngpts, nmaps, map);
    }
    host_fft_3d(work, npts, 1);
    for (size_t i = 0; i < n; i++) {
      out[d][i] = work[2 * i];
    }
  }
  free(work);
}

static void host_div3(const double *in[3], double *out, const int *map,
                      const double *g, const int *npts, const int ngpts,
                      const int nmaps, const double scale) {
  const size_t n = (size_t)npts[0] * npts[1] * npts[2];
  double *work = host_buffer(2 * n);
  double *acc = host_buffer(2 * (size_t)ngpts);
  for (int d = 0; d < 3; d++) {
    for (size_t i = 0; i < n; i++) {
      work[2 * i] = in[d][i];
      work[2 * i + 1] = 0.0;
    }
    host_fft_3d(work, npts, -1);
    for (int i = 0; i < ngpts; i++) {
      pw_fused_gather_deriv_acc_elem(acc, work, g, d, scale, i, map);
    }
  }
  // Consume all three inputs before writing out, which may alias in[0].
  host_sfffc(acc, out, map, npts, ngpts, nmaps, 1.0);
  free(acc);
  free(work);
}

static void host_filter(double *data, const int *map, const int *npts,
                        const int ngpts, const int nmaps, const double scale) {
  double *compact = host_buffer(2 * (size_t)ngpts);
  host_cfffg(data, compact, map, npts, ngpts, scale);
  host_sfffc(compact, data, map, npts, ngpts, nmaps, 1.0);
  free(compact);
}

// The one/two-dimensional real/complex transforms used by distributed grids.
static void host_real_fft(const double *in, double *out, const int *npts,
                          const int sign, const int dimensions) {
  const size_t n = (size_t)npts[0] * npts[1] * npts[2];
  double *a = host_buffer(2 * n), *b = host_buffer(2 * n);
  if (sign < 0) {
    for (size_t i = 0; i < n; i++) {
      a[2 * i] = in[i];
    }
    host_fft_1d(-1, npts[2], npts[0] * npts[1], a, b);
    if (dimensions == 2) {
      host_fft_1d(-1, npts[1], npts[0] * npts[2], b, a);
    }
    memcpy(out, dimensions == 2 ? a : b, 2 * sizeof(double) * n);
  } else {
    memcpy(a, in, 2 * sizeof(double) * n);
    if (dimensions == 2) {
      host_fft_1d(1, npts[1], npts[0] * npts[2], a, b);
      memcpy(a, b, 2 * sizeof(double) * n);
    }
    host_fft_1d(1, npts[2], npts[0] * npts[1], a, b);
    for (size_t i = 0; i < n; i++) {
      out[i] = b[2 * i];
    }
  }
  free(b);
  free(a);
}

static void host_fg(const double *in, double *out, const int *map,
                    const int n, const int m, const int ngpts, const double scale) {
  double *work = host_buffer(2 * (size_t)n * m);
  host_fft_1d(-1, n, m, in, work);
  host_gather(out, work, map, ngpts, scale);
  free(work);
}

static void host_sf(const double *in, double *out, const int *map,
                    const int n, const int m, const int ngpts, const int nmaps,
                    const double scale) {
  double *work = host_buffer(2 * (size_t)n * m);
  host_scatter(work, in, map, ngpts, nmaps, scale);
  host_fft_1d(1, n, m, work, out);
  free(work);
}

#endif
