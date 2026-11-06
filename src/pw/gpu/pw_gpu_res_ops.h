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

#include <math.h>
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

/*******************************************************************************
 * \brief Natural FFT order of XC (stage 4). On a full grid (every FFT point a
 *        G vector, g_hatmap a permutation) the G-space arrays are kept at their
 *        FFT positions p = i0 + n0*(i1 + n1*i2). The element operations are
 *        those of pw_fused_scatter_deriv_elem / pw_fused_gather_deriv_acc_elem
 *        with the map removed, so every value is bitwise the same.
 *
 *        g(idir) at FFT point p: mode 1 = axis tables gt[0,n0) | gt[n0,n0+n1) |
 *        gt[n0+n1, n0+n1+n2) indexed by the FFT index along each axis (the g
 *        vectors of an orthorhombic cell are separable), mode 2 = full copy
 *        gt(3, nr) in FFT order.
 ******************************************************************************/
// Grid points fit in 31 bits (the callers check nr <= INT_MAX / 2), so the
// axis indices use 32-bit divisions, which are much cheaper on the device.
PW_GPU_RES_HD double pw_nat_gd(const double *gt, const int mode, const size_t n0,
                               const size_t n1, const size_t p, const int idir) {
  if (mode == 2) return gt[3 * p + (size_t)idir];
  const unsigned int q = (unsigned int)p, m0 = (unsigned int)n0, m1 = (unsigned int)n1;
  if (idir == 0) return gt[q % m0];
  if (idir == 1) return gt[n0 + (q / m0) % m1];
  return gt[n0 + n1 + q / (m0 * m1)];
}

// pw_fused_scatter_deriv_elem at FFT point p (nmaps = 1).
PW_GPU_RES_HD void pw_nat_deriv_elem(double *c, const double *zin,
                                     const double gd, const double scale,
                                     const size_t p) {
  PW_GPU_NO_CONTRACT
  const double tr = PW_GPU_DMUL(zin[2 * p], gd);
  const double ti = PW_GPU_DMUL(zin[2 * p + 1], gd);
  const double dr = -ti;
  const double di = tr;
  c[2 * p] = PW_GPU_DMUL(scale, dr);
  c[2 * p + 1] = PW_GPU_DMUL(scale, di);
}

// pw_fused_gather_deriv_acc_elem at FFT point p.
PW_GPU_RES_HD void pw_nat_deriv_acc_elem(double *acc, const double *c,
                                         const double gd, const double scale,
                                         const size_t p) {
  PW_GPU_NO_CONTRACT
  const double pr = PW_GPU_DMUL(scale, c[2 * p]);
  const double pi = PW_GPU_DMUL(scale, c[2 * p + 1]);
  const double tr = PW_GPU_DMUL(pr, gd);
  const double ti = PW_GPU_DMUL(pi, gd);
  acc[2 * p] = PW_GPU_DADD(acc[2 * p], -ti);
  acc[2 * p + 1] = PW_GPU_DADD(acc[2 * p + 1], tr);
}

// Conjugate partner of FFT position p: index (-i) mod n on every axis. A real
// field f satisfies FFT[f](mirror(p)) = conj FFT[f](p) exactly (also on the
// Nyquist planes of even axes, which are their own mirror).
PW_GPU_RES_HD size_t pw_nat_mirror(const size_t p, const size_t n0,
                                   const size_t n1, const size_t n2) {
  const unsigned int q = (unsigned int)p, m0 = (unsigned int)n0;
  const unsigned int m1 = (unsigned int)n1, m2 = (unsigned int)n2;
  const unsigned int x = q % m0, r = q / m0, y = r % m1, z = r / m1;
  const unsigned int mx = x ? m0 - x : 0, my = y ? m1 - y : 0, mz = z ? m2 - z : 0;
  return (size_t)mx + (size_t)m0 * ((size_t)my + (size_t)m1 * (size_t)mz);
}

/*******************************************************************************
 * \brief Two XC gradients in one inverse FFT (stage 5). With F_d = i g_d rho
 *        and F^h(p) = (F(p) + conj F(mirror(p))) / 2 its Hermitian part,
 *        Z = Fa^h + i Fb^h gives IFFT[Z] = Re IFFT[Fa] + i Re IFFT[Fb]: the
 *        real parts that complex_to_real takes of each gradient alone, for
 *        any rho (no symmetry of rho_g is assumed). zout must not alias rho.
 ******************************************************************************/
PW_GPU_RES_HD void pw_nat_grad2_elem(double *zout, const double *rho, const double *gt,
                                     const int mode, const size_t n0, const size_t n1,
                                     const size_t n2, const int a, const int b,
                                     const size_t p) {
  PW_GPU_NO_CONTRACT
  const size_t m = pw_nat_mirror(p, n0, n1, n2);
  const double ga = pw_nat_gd(gt, mode, n0, n1, p, a), gb = pw_nat_gd(gt, mode, n0, n1, p, b);
  const double gam = pw_nat_gd(gt, mode, n0, n1, m, a), gbm = pw_nat_gd(gt, mode, n0, n1, m, b);
  const double rr = rho[2 * p], ri = rho[2 * p + 1], sr = rho[2 * m], si = rho[2 * m + 1];
  // F_d(q) = (-g_d(q) rho_i(q), g_d(q) rho_r(q)), conj F_d(m) = (-g_d(m) s_i, -g_d(m) s_r)
  const double far = 0.5 * (-PW_GPU_DMUL(ga, ri) - PW_GPU_DMUL(gam, si));
  const double fai = 0.5 * (PW_GPU_DMUL(ga, rr) - PW_GPU_DMUL(gam, sr));
  const double fbr = 0.5 * (-PW_GPU_DMUL(gb, ri) - PW_GPU_DMUL(gbm, si));
  const double fbi = 0.5 * (PW_GPU_DMUL(gb, rr) - PW_GPU_DMUL(gbm, sr));
  zout[2 * p] = far - fbi;
  zout[2 * p + 1] = PW_GPU_DADD(fai, fbr);
}

/*******************************************************************************
 * \brief Two divergence terms from one forward FFT (stage 5). W = FFT[h_a +
 *        i h_b] of two real fields gives H_a = (W(p) + conj W(m)) / 2 and
 *        H_b = (W(p) - conj W(m)) / 2i exactly (m = mirror(p)); acc(p) is set
 *        to (accumulate = 0) or incremented by i scale (g_a H_a + g_b H_b).
 *        acc must not alias w.
 ******************************************************************************/
PW_GPU_RES_HD void pw_nat_div2_elem(double *acc, const double *w, const double *gt,
                                    const int mode, const size_t n0, const size_t n1,
                                    const size_t n2, const int a, const int b,
                                    const double scale, const int accumulate,
                                    const size_t p) {
  PW_GPU_NO_CONTRACT
  const size_t m = pw_nat_mirror(p, n0, n1, n2);
  const double wr = w[2 * p], wi = w[2 * p + 1], vr = w[2 * m], vi = w[2 * m + 1];
  const double har = 0.5 * PW_GPU_DADD(wr, vr), hai = 0.5 * (wi - vi);
  const double hbr = 0.5 * PW_GPU_DADD(wi, vi), hbi = 0.5 * (vr - wr);
  const double ga = pw_nat_gd(gt, mode, n0, n1, p, a), gb = pw_nat_gd(gt, mode, n0, n1, p, b);
  const double sr = PW_GPU_DADD(PW_GPU_DMUL(ga, har), PW_GPU_DMUL(gb, hbr));
  const double si = PW_GPU_DADD(PW_GPU_DMUL(ga, hai), PW_GPU_DMUL(gb, hbi));
  const double tr = -PW_GPU_DMUL(scale, si), ti = PW_GPU_DMUL(scale, sr);
  if (accumulate) {
    acc[2 * p] = PW_GPU_DADD(acc[2 * p], tr);
    acc[2 * p + 1] = PW_GPU_DADD(acc[2 * p + 1], ti);
  } else {
    acc[2 * p] = tr;
    acc[2 * p + 1] = ti;
  }
}

/*******************************************************************************
 * \brief Density and its z derivative from one inverse FFT (stage 6). With
 *        F = i g_z rho, Z = rho^h + i F^h gives IFFT[Z] = Re IFFT[rho] +
 *        i Re IFFT[F]: rho_r and the third XC gradient, each as
 *        complex_to_real takes it of its own transform, for any rho.
 *        zout must not alias rho.
 ******************************************************************************/
PW_GPU_RES_HD void pw_nat_rho_dz_elem(double *zout, const double *rho, const double *gt,
                                      const int mode, const size_t n0, const size_t n1,
                                      const size_t n2, const size_t p) {
  PW_GPU_NO_CONTRACT
  const size_t m = pw_nat_mirror(p, n0, n1, n2);
  const double gz = pw_nat_gd(gt, mode, n0, n1, p, 2), gzm = pw_nat_gd(gt, mode, n0, n1, m, 2);
  const double rr = rho[2 * p], ri = rho[2 * p + 1], sr = rho[2 * m], si = rho[2 * m + 1];
  const double har = 0.5 * PW_GPU_DADD(rr, sr), hai = 0.5 * (ri - si);
  const double fr = 0.5 * (-PW_GPU_DMUL(gz, ri) - PW_GPU_DMUL(gzm, si));
  const double fi = 0.5 * (PW_GPU_DMUL(gz, rr) - PW_GPU_DMUL(gzm, sr));
  zout[2 * p] = har - fi;
  zout[2 * p + 1] = PW_GPU_DADD(hai, fr);
}

// Hartree (stage 5): rho_tot = core + rho_g (pw_axpy onto the copied core) and
// out = rho_tot * w (pw_res_cmul_elem) in one pass, bitwise as the three
// separate kernels. core may alias rho_tot, rg may alias out.
PW_GPU_RES_HD void pw_res_tot_cmul_elem(double *rho_tot, double *out, const double *core,
                                        const double *rg, const double *w,
                                        const int width, const size_t i) {
  PW_GPU_NO_CONTRACT
  const double tr = PW_GPU_DADD(core[2 * i], rg[2 * i]);
  const double ti = PW_GPU_DADD(core[2 * i + 1], rg[2 * i + 1]);
  rho_tot[2 * i] = tr;
  rho_tot[2 * i + 1] = ti;
  pw_res_cmul_elem(out, rho_tot, w, width, i);
}

// Bit patterns differ (also +0 versus -0); the g vectors contain no NaN.
PW_GPU_RES_HD int pw_res_bits_differ(const double a, const double b) {
  return !(a == b) || (signbit(a) != 0) != (signbit(b) != 0);
}

#endif

// EOF
