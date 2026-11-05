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

void res_fixture_end(void) {
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
  pw_gpu_finalize();
#endif
}
