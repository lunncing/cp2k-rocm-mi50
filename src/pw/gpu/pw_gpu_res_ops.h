/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

/*******************************************************************************
 * \brief Element operations of the device-resident PW chain (density sum,
 *        Hartree/DDAPC, potential distribution) and the sampled fingerprints
 *        that tie a device copy to its host array.
 *
 *        Include after pw_gpu_fused_ops.h (DMUL/DADD/NO_CONTRACT). The same
 *        functions are compiled into the HIP/CUDA kernels, the host code and
 *        the mock test harness, so the sample positions agree everywhere.
 ******************************************************************************/
#ifndef PW_GPU_RES_OPS_H
#define PW_GPU_RES_OPS_H

#include <stddef.h>

#if !defined(PW_GPU_FUSED_OPS_H)
#error "Include pw_gpu_fused_ops.h before pw_gpu_res_ops.h"
#endif

#if defined(__HIPCC__) || defined(__CUDACC__)
#define PW_GPU_RES_HD static __host__ __device__ inline
#else
#define PW_GPU_RES_HD static inline
#endif

// Fingerprint: the first PREFIX elements (the low-|G| shells of sorted
// G-space arrays) plus one pseudo-random element in each of STRIDED equal
// buckets covering the rest of the array.
#define PW_GPU_RES_PREFIX_SAMPLES 1024
#define PW_GPU_RES_STRIDED_SAMPLES 16384

PW_GPU_RES_HD size_t pw_res_sample_count(const size_t n) {
  const size_t p = n < PW_GPU_RES_PREFIX_SAMPLES ? n : PW_GPU_RES_PREFIX_SAMPLES;
  const size_t rest = n - p;
  return p + (rest < PW_GPU_RES_STRIDED_SAMPLES ? rest : PW_GPU_RES_STRIDED_SAMPLES);
}

// Element index of sample k < pw_res_sample_count(n). All products stay
// below 2^15 * 2^31 and therefore fit into 64 bits.
PW_GPU_RES_HD size_t pw_res_sample_index(size_t k, const size_t n) {
  const size_t p = n < PW_GPU_RES_PREFIX_SAMPLES ? n : PW_GPU_RES_PREFIX_SAMPLES;
  if (k < p) {
    return k;
  }
  k -= p;
  const size_t rest = n - p;
  const size_t buckets =
      rest < PW_GPU_RES_STRIDED_SAMPLES ? rest : PW_GPU_RES_STRIDED_SAMPLES;
  const unsigned long long lo = (unsigned long long)k * rest / buckets;
  const unsigned long long hi = (unsigned long long)(k + 1) * rest / buckets;
  const unsigned long long hash = ((unsigned long long)k * 2654435761ULL) >> 7;
  return p + (size_t)(lo + hash % (hi - lo));
}

/*******************************************************************************
 * \brief Fortran complex multiply out = a * (c + i d), each product and sum
 *        rounded separately. width = 1 stores only c (d = +0, verified when
 *        the influence function is registered), width = 2 stores (c, d).
 *        out may alias a.
 ******************************************************************************/
PW_GPU_RES_HD void pw_res_cmul_elem(double *out, const double *a,
                                    const double *w, const int width,
                                    const size_t i) {
  PW_GPU_NO_CONTRACT
  const double ar = a[2 * i], ai = a[2 * i + 1];
  const double c = w[(size_t)width * i];
  const double d = width == 2 ? w[2 * i + 1] : 0.0;
  out[2 * i] = PW_GPU_DMUL(ar, c) - PW_GPU_DMUL(ai, d);
  out[2 * i + 1] = PW_GPU_DADD(PW_GPU_DMUL(ar, d), PW_GPU_DMUL(ai, c));
}

/*******************************************************************************
 * \brief One term of Re(sum conj(a) b) with interleaved complex arrays.
 ******************************************************************************/
PW_GPU_RES_HD double pw_res_cdot_term(const double *a, const double *b,
                                      const size_t i) {
  PW_GPU_NO_CONTRACT
  return PW_GPU_DADD(PW_GPU_DMUL(a[2 * i], b[2 * i]),
                     PW_GPU_DMUL(a[2 * i + 1], b[2 * i + 1]));
}

#endif

// EOF
