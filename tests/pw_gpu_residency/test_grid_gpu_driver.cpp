/* SPDX-License-Identifier: GPL-2.0-or-later */
// Compile the production GPU task-list orchestration against a tiny context.
// Arithmetic is a known per-level fill/sum; copies and stream ordering use the
// same checked runtime as the PW tests, including asynchronous lifetime checks.
#include <cassert>
#include <cstddef>
#include <cmath>
#include <cstdio>
extern "C" {
#include "mock_offload.h"
#include "offload_buffer.h"
}
enum grid_func { GRID_FUNC_AB };
enum { GRID_BACKEND_GPU, GRID_COLLOCATE_ORTHO, GRID_COLLOCATE_GENERAL,
       GRID_INTEGRATE_ORTHO, GRID_INTEGRATE_GENERAL };
typedef void grid_gpu_task_list;
static void grid_library_counter_add(int, int, int, int) {}
namespace rocm_backend {
struct buffer {
  double *host = nullptr, *device = nullptr;
  size_t n = 0;
  void associate(double *h, double *d, size_t count) { host = h; device = d; n = count; }
  void zero(int s) { if (n) offloadMemsetAsync(device, 0, n * sizeof(double), s); }
  void copy_to_host(int s) { if (n) offloadMemcpyAsyncDtoH(host, device, n * sizeof(double), s); }
  void copy_to_gpu(int s) { if (n) offloadMemcpyAsyncHtoD(device, host, n * sizeof(double), s); }
  void copy_associated_host_to_gpu(int s) { copy_to_gpu(s); }
  void copy_gpu_to_associated_host(int s) { copy_to_host(s); }
  bool is_orthogonal() { return true; }
  void resize(size_t) { assert(false && "forces unused in this fixture"); }
  void copy_from_gpu(double *, int) { assert(false); }
};
struct context_info {
  int nblocks = 1, ntasks = 1, nlevels = 2, natoms = 1, main_stream = 1;
  int level_streams[2] = {2, 2}, number_of_tasks_per_level_[2] = {1, 1};
  int stats[2][20] = {{0}};
  bool calculate_forces = false, calculate_virial = false, compute_tau = false;
  buffer grid_[2], pab_block_, coef_dev_, hab_block_, forces_, virial_;
  void verify_checksum() {}
  void set_device() {}
  void synchronize(int s) { offloadStreamSynchronize(s); }
  void calculate_all_coefficients(grid_func, int *diff) { *diff = -1; }
  void collocate_one_grid_level(int l, grid_func, int *) {
    for (size_t i = 0; i < grid_[l].n; i++) grid_[l].device[i] = l + 1.0 + 0.01 * i;
  }
  void integrate_one_grid_level(int l, int *diff) {
    // Model the real context's dependency on the zeroed HAB allocation.
    offloadStreamSynchronize(main_stream);
    offloadStreamSynchronize(level_streams[l]);
    for (size_t i = 0; i < grid_[l].n; i++) hab_block_.device[0] += grid_[l].device[i];
    *diff = -1;
  }
  void compute_hab_coefficients() {}
};
}
#include "grid_gpu_calls.inc"

int main() {
  offloadStream_t a, b;
  offloadStreamCreate(&a); offloadStreamCreate(&b);
  rocm_backend::context_info ctx;
  offload_buffer *grids[2] = {nullptr, nullptr}, *pab = nullptr, *hab = nullptr;
  offload_create_buffer(1, &pab); offload_create_buffer(1, &hab);
  pab->host_buffer[0] = 2;
  for (int l = 0; l < 2; l++) {
    offload_create_buffer(12, &grids[l]);
    for (int i = 0; i < 12; i++) grids[l]->host_buffer[i] = -123;
    offload_grid_prepare(grids[l], 1);
  }
  mock_reset_trace();
  grid_gpu_collocate_task_list(&ctx, GRID_FUNC_AB, 2, pab, grids);
  assert(mock_download_bytes == 0);
  for (int l = 0; l < 2; l++) {
    assert(offload_grid_device(grids[l], 12 * sizeof(double)) == grids[l]->device_buffer);
    for (int i = 0; i < 12; i++) assert(grids[l]->host_buffer[i] == -123);
  }
  const offload_buffer *inputs[2] = {grids[0], grids[1]};
  mock_reset_trace();
  grid_gpu_integrate_task_list(&ctx, false, 2, nullptr, inputs, hab, nullptr, nullptr);
  assert(mock_upload_bytes == 0);
  const double expected = 36 + 0.02 * 66;
  assert(hab->host_buffer[0] == expected || std::abs(hab->host_buffer[0] - expected) < 1e-12);
  // Borrow a smaller PW view than the owned Grid capacity. The association
  // must use the view size and must not upload the stale Grid host storage.
  double *alias;
  offloadMalloc((void **)&alias, 6 * sizeof(double));
  double values[6] = {4, 4, 4, 4, 4, 4};
  offloadMemcpyAsyncHtoD(alias, values, sizeof(values), a);
  offloadStreamSynchronize(a);
  assert(offload_grid_bind(grids[0], alias, sizeof(values)));
  mock_reset_trace();
  grid_gpu_integrate_task_list(&ctx, false, 2, nullptr, inputs, hab, nullptr, nullptr);
  assert(mock_upload_bytes == 0 && ctx.grid_[0].n == 6 && ctx.grid_[0].device == alias);
  assert(std::abs(hab->host_buffer[0] - (48 + 0.01 * 66)) < 1e-12);
  offload_grid_prepare(grids[0], 0);
  offloadFree(alias);
  // The normal eager producer retains its host contract.
  for (int l = 0; l < 2; l++) offload_grid_prepare(grids[l], 0);
  mock_reset_trace();
  grid_gpu_collocate_task_list(&ctx, GRID_FUNC_AB, 2, pab, grids);
  assert(mock_download_bytes == 24 * sizeof(double));
  for (int l = 0; l < 2; l++) {
    assert(!offload_grid_device(grids[l], 12 * sizeof(double)));
    for (int i = 0; i < 12; i++) assert(grids[l]->host_buffer[i] == l + 1.0 + 0.01 * i);
  }
  mock_reset_trace();
  grid_gpu_integrate_task_list(&ctx, false, 2, nullptr, inputs, hab, nullptr, nullptr);
  assert(mock_upload_bytes == 24 * sizeof(double));
  for (auto *g : grids) offload_free_buffer(g);
  offload_free_buffer(pab); offload_free_buffer(hab);
  offloadStreamDestroy(a); offloadStreamDestroy(b);
  assert(mock_allocated == 0);
  puts("PASS production Grid GPU orchestration: deferred collocation and borrowed integration");
}
