/* SPDX-License-Identifier: GPL-2.0-or-later */
#include "mock_offload.h"
#include "offload_buffer.h"
#include <assert.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum grid_func { GRID_FUNC_AB };
enum { GRID_BACKEND_REF, GRID_BACKEND_CPU, GRID_BACKEND_DGEMM, GRID_BACKEND_GPU };
typedef void grid_task_list;
typedef struct {
  bool empty;
  int backend, nlevels, npts_local[1][3];
  void *ref, *cpu, *dgemm, *gpu;
} grid_task_list_internal;
typedef struct { bool validate; } config_type;
static config_type config;
static config_type grid_library_get_config(void) { return config; }
static void grid_ref_collocate_task_list(void *p, enum grid_func f, int n,
                                         const offload_buffer *pab, offload_buffer **grids) {
  (void)p; (void)f; (void)pab;
  for (int l = 0; l < n; l++)
    for (size_t i = 0; i < grids[l]->size / sizeof(double); i++) grids[l]->host_buffer[i] = 2.0;
}
#define grid_cpu_collocate_task_list grid_ref_collocate_task_list
#define grid_dgemm_collocate_task_list grid_ref_collocate_task_list
static void grid_ref_integrate_task_list(void *p, bool tau, int natoms, int n,
    const offload_buffer *pab, const offload_buffer **grids, offload_buffer *hab,
    double (*forces)[3], double (*virial)[3]) {
  (void)p; (void)tau; (void)pab;
  hab->host_buffer[0] = 0;
  for (int l = 0; l < n; l++)
    for (size_t i = 0; i < grids[l]->size / sizeof(double); i++)
      hab->host_buffer[0] += grids[l]->host_buffer[i];
  if (forces) memset(forces, 0, (size_t)natoms * 3 * sizeof(double));
  if (virial) memset(virial, 0, 9 * sizeof(double));
}
#define grid_cpu_integrate_task_list grid_ref_integrate_task_list
#define grid_dgemm_integrate_task_list grid_ref_integrate_task_list
static void grid_gpu_collocate_task_list(void *p, enum grid_func f, int n,
                                         const offload_buffer *pab, offload_buffer **grids) {
  (void)p; (void)f; (void)pab;
  for (int l = 0; l < n; l++) {
    double values[12];
    for (int i = 0; i < 12; i++) values[i] = 2;
    offloadMemcpyAsyncHtoD(grids[l]->device_buffer, values, sizeof(values), 2);
    if (!grids[l]->grid_defer_host)
      offloadMemcpyAsyncDtoH(grids[l]->host_buffer, grids[l]->device_buffer, sizeof(values), 2);
    offloadStreamSynchronize(2);
    grids[l]->grid_device_valid = grids[l]->grid_defer_host;
    grids[l]->grid_defer_host = 0;
    grids[l]->grid_bytes = sizeof(values);
  }
}
static void grid_gpu_integrate_task_list(void *p, bool tau, int n,
    const offload_buffer *pab, const offload_buffer **grids, offload_buffer *hab,
    double *forces, double *virial) {
  (void)p; (void)tau; (void)pab;
  hab->host_buffer[0] = 0;
  for (int l = 0; l < n; l++) {
    const double *data = offload_grid_device(grids[l], grids[l]->size);
    if (!data) data = grids[l]->host_buffer;
    for (int i = 0; i < 12; i++) hab->host_buffer[0] += data[i];
  }
  if (forces) memset(forces, 0, 3 * sizeof(double));
  if (virial) memset(virial, 0, 9 * sizeof(double));
}
#include "grid_dispatch_calls.inc"

int main(void) {
  offloadStream_t a, b;
  offloadStreamCreate(&a); offloadStreamCreate(&b);
  grid_task_list_internal task = {.backend = GRID_BACKEND_GPU, .nlevels = 1,
                                  .npts_local = {{3, 2, 2}}};
  const int dims[1][3] = {{3, 2, 2}};
  offload_buffer *grid = NULL, *hab = NULL;
  offload_create_buffer(12, &grid); offload_create_buffer(1, &hab);
  double *alias;
  offloadMalloc((void **)&alias, grid->size);
  double values[12];
  for (int i = 0; i < 12; i++) values[i] = 4;
  offloadMemcpyAsyncHtoD(alias, values, sizeof(values), a); offloadStreamSynchronize(a);
  offload_buffer *grids[] = {grid};
  const offload_buffer *inputs[] = {grid};
  double forces[1][3], virial[3][3];
  for (int backend = GRID_BACKEND_REF; backend <= GRID_BACKEND_GPU; backend++) {
    task.backend = backend;
    for (int validate = 0; validate <= 1; validate++) {
      config.validate = validate;
      grid->host_buffer[0] = -1;
      offload_grid_prepare(grid, 1);
      grid_collocate_task_list(&task, GRID_FUNC_AB, 1, dims, NULL, grids);
      const bool lazy = backend == GRID_BACKEND_GPU && !validate;
      assert(grid->grid_device_valid == lazy && !grid->grid_defer_host);
      assert(grid->host_buffer[0] == (lazy ? -1 : 2));
      // A stale owned allocation cannot substitute for a borrowed PW view.
      assert(offload_grid_bind(grid, alias, grid->size));
      grid_integrate_task_list(&task, false, 1, 1, dims, grid, inputs, hab, forces, virial);
      assert(hab->host_buffer[0] == 48);
      assert(!grid->grid_device_valid && !grid->grid_device_view);
    }
  }
  config.validate = false;
  task.empty = true;
  offload_grid_prepare(grid, 1);
  grid_collocate_task_list(&task, GRID_FUNC_AB, 1, dims, NULL, grids);
  assert(grid->host_buffer[0] == 0 && !grid->grid_device_valid && !grid->grid_defer_host);
  for (int empty = 0; empty <= 1; empty++) {
    assert(offload_grid_bind(grid, alias, grid->size));
    mock_reset_trace();
    grid_integrate_task_list(empty ? &task : NULL, false, 1, 1, dims, grid, inputs, hab, forces, virial);
    assert(!grid->grid_device_view && mock_download_bytes == 0);
  }
  offloadFree(alias);
  offload_free_buffer(grid); offload_free_buffer(hab);
  offloadStreamDestroy(a); offloadStreamDestroy(b);
  assert(mock_allocated == 0);
  puts("PASS Grid dispatch: CPU/GPU/validation, empty tasks and borrowed-view retirement");
}
