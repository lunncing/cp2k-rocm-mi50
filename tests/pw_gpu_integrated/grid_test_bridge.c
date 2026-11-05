/* SPDX-License-Identifier: GPL-2.0-or-later */
#include "pw_gpu_internal.c"

void test_gpu_begin(int limited) {
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
  extern size_t mock_limit;
  pw_gpu_init();
  release_work_buffers();
  max_gpu_memory_bytes = mock_limit = limited ? 1 : 0;
#else
  (void)limited;
#endif
}

void test_gpu_end(void) {
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
  extern size_t mock_allocated;
  extern void mock_assert_idle(void);
  mock_assert_idle();
  pw_gpu_finalize();
  assert(mock_allocated == 0);
#endif
}

int test_kernel_calls(int kind) {
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
  extern int mock_grid_calls, mock_hartree_calls, mock_pbe_calls;
  return kind == 0 ? mock_grid_calls : kind == 1 ? mock_hartree_calls : mock_pbe_calls;
#else
  (void)kind;
  return 0;
#endif
}
