/* SPDX-License-Identifier: GPL-2.0-or-later */
#include <stddef.h>
#include "pw_gpu_internal.c"

// Seed the same borrowed slots as the density FFT without depending on a DFT
// in the binding tests. The numerical density/FFT chain is tested separately.
void res_fixture_begin(double *rho, double *g, size_t n) {
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
  pw_gpu_init();
  if (!opt_res_density) return;
  assert(pw_gpu_ensure_xc_c(4 * sizeof(double) * n));
  double *small = res_small(2 * pw_res_sample_count(n));
  assert(small != NULL);
  assert(res_sample_buffer(&res[RES_RHO_G], 2 * pw_res_sample_count(n)));
  assert(res_sample_buffer(&res[RES_RHO_R], pw_res_sample_count(n)));
  offloadMemcpyAsyncHtoD(xc_dev_hi, rho, sizeof(double) * n, stream);
  offloadMemcpyAsyncHtoD(xc_dev, g, 2 * sizeof(double) * n, stream);
  offloadStreamSynchronize(stream);
  memset(rho, 0, sizeof(double) * n);
  memset(g, 0, 2 * sizeof(double) * n);
  res_register_lazy(RES_RHO_R, rho, n, 1, xc_dev_hi, small);
  res_register_lazy(RES_RHO_G, g, n, 2, xc_dev, small);
  offloadStreamSynchronize(stream);
#else
  (void)rho; (void)g; (void)n;
#endif
}

void res_fixture_vxc(double *vxc, size_t n) {
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
  if (!opt_res_pot) return;
  res_invalidate_borrowed();
  offloadMemcpyAsyncHtoD(xc_dev, vxc, sizeof(double) * n, stream);
  offloadStreamSynchronize(stream);
  res_register(RES_VXC, vxc, n, 1, 1, xc_dev);
#else
  (void)vxc; (void)n;
#endif
}

void res_fixture_spin_b(double *rho, double *g, size_t n) {
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
  if (!opt_res_density || !opt_uks) return;
  assert(ensure_uks_buffer(&uks_density_dev, &allocated_uks_density_size, 4 * n * sizeof(double)));
  double *small = res_small(2 * pw_res_sample_count(n));
  assert(small != NULL);
  assert(res_sample_buffer(&res[RES_RHO_G_B], 2 * pw_res_sample_count(n)));
  assert(res_sample_buffer(&res[RES_RHO_R_B], pw_res_sample_count(n)));
  offloadMemcpyAsyncHtoD(uks_density_dev, g, 2 * n * sizeof(double), stream);
  offloadMemcpyAsyncHtoD(uks_density_dev + 2 * n, rho, n * sizeof(double), stream);
  offloadStreamSynchronize(stream);
  memset(g, 0, 2 * n * sizeof(double));
  memset(rho, 0, n * sizeof(double));
  res_register_lazy(RES_RHO_G_B, g, n, 2, uks_density_dev, small);
  res_register_lazy(RES_RHO_R_B, rho, n, 1, uks_density_dev + 2 * n, small);
  offloadStreamSynchronize(stream);
  protected_uks_buffer = false;
#else
  (void)rho; (void)g; (void)n;
#endif
}

void res_fixture_vxc_pair(double *va, double *vb, size_t n) {
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
  if (!opt_res_pot || !opt_uks) {
    res_fixture_vxc(va, n);
    return;
  }
  assert(ensure_uks_buffer(&uks_density_dev, &allocated_uks_density_size, 4 * n * sizeof(double)));
  res_invalidate_borrowed();
  offloadMemcpyAsyncHtoD(uks_density_dev, va, n * sizeof(double), stream);
  offloadMemcpyAsyncHtoD(xc_dev, vb, n * sizeof(double), stream);
  offloadStreamSynchronize(stream);
  res_register(RES_VXC, va, n, 1, 1, uks_density_dev);
  res_register(RES_VXC_B, vb, n, 1, 1, xc_dev);
  protected_uks_buffer = false;
#else
  (void)va; (void)vb; (void)n;
#endif
}

void res_fixture_end(void) {
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
  pw_gpu_finalize();
#endif
}
