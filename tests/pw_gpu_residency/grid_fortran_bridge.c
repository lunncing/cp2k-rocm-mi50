/* SPDX-License-Identifier: GPL-2.0-or-later */
#include "pw_gpu_internal.c"
#include "offload_buffer.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

void grid_fixture_begin(void) {
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
  pw_gpu_init();
  pw_gpu_res_set_lazy_c(1);
  pw_gpu_res_defer_density_c(1);
#endif
}
void grid_fixture_collocate(offload_buffer *b) {
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
  if (!b->grid_defer_host) return;
  offloadMemcpyAsyncHtoD(b->device_buffer, b->host_buffer, b->size, stream_copy);
  offloadStreamSynchronize(stream_copy);
  b->grid_device_valid = 1; b->grid_bytes = b->size;
  b->grid_defer_host = 0;
  for (size_t i = 0; i < b->size / sizeof(double); i++) b->host_buffer[i] = -321;
#else
  (void)b;
#endif
}
void grid_fixture_potential(double *host, size_t n) {
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
  if (!opt_res_pot) return;
  assert(pw_gpu_ensure_xc_c(4 * n * sizeof(double)));
  res_invalidate_borrowed();
  offloadMemcpyAsyncHtoD(xc_dev, host, n * sizeof(double), stream);
  offloadStreamSynchronize(stream);
  res_register(RES_VXC, host, n, 1, 1, xc_dev);
  res[RES_VXC].vtot = true;
  res[RES_VXC].dev_authoritative = true;
  for (size_t i = 0; i < n; i++) host[i] = -654;
#else
  (void)host; (void)n;
#endif
}
void grid_fixture_end(void) {
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
  pw_gpu_res_set_lazy_c(0);
  pw_gpu_finalize();
#endif
}
