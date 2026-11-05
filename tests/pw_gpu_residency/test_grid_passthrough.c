/* SPDX-License-Identifier: GPL-2.0-or-later */
#define main original_residency_main
#include "test_res.c"
#undef main
#include "offload_buffer.h"

static int density_device(double *const *host, void *const *device, double *rg, double *rr) {
  void *mp[NLEV], *gp[NLEV], *hp[NLEV];
  int npts[3 * NLEV], ids[NLEV], nm[NLEV];
  double scale[NLEV];
  for (int l = 0; l < NLEV; l++) {
    hp[l] = host[l]; mp[l] = map_[l]; gp[l] = gidx_[l];
    memcpy(npts + 3 * l, dims[l], sizeof(dims[l]));
    ids[l] = l + 1; nm[l] = nmaps_; scale[l] = 1.0 / nr_[l];
  }
  pw_gpu_res_defer_density_c(1);
  return pw_gpu_res_density_c(NLEV, hp, npts, ng_, nm, ids, mp, gp, scale, dims[0],
                             ng_[0], nmaps_, 1, map_[0], rg, rr, device);
}

static int potential_device(const double *v, double *const *host, void **device) {
  void *mp[NLEV], *gp[NLEV], *hp[NLEV];
  int npts[3 * NLEV], ids[NLEV], nm[NLEV];
  const double scale[NLEV] = {1.0, 8.0, 0.5};
  for (int l = 0; l < NLEV; l++) {
    hp[l] = host[l]; mp[l] = map_[l]; gp[l] = gidx_[l];
    memcpy(npts + 3 * l, dims[l], sizeof(dims[l]));
    ids[l] = l + 1; nm[l] = nmaps_;
  }
  return pw_gpu_res_potential_c(v, NLEV, hp, npts, ng_, nm, ids, mp, gp, scale, 0,
                               1.0 / nr_[0], device);
}

int main(void) {
  grids(1);
  start(0);
  assert(pw_gpu_ensure_buffer_c(2 * sizeof(double) * (size_t)nr_[0], 0, false));
  grid_cache_entry *cache_entry = grid_cache_lookup_or_upload(7, map_[0], ng_[0], 1);
  assert(cache_entry != NULL && grid_cache_get_g(cache_entry, g_fine) != NULL);
  offloadStreamSynchronize(stream);
  const size_t core_bytes = 2 * sizeof(double) * (size_t)ng_[0];
  max_gpu_memory_bytes = mock_limit = current_gpu_allocated_bytes;
  protected_grid_entry = cache_entry;
  assert(!res_ensure_owned(RES_RHO_CORE, core_bytes));
  protected_grid_entry = NULL;
  assert(res_ensure_owned(RES_RHO_CORE, core_bytes));
  assert(cache_entry->grid_id == 0);
  max_gpu_memory_bytes = mock_limit = 0;
  res_free_owned(RES_RHO_CORE);
  offload_buffer *buffers[NLEV] = {NULL};
  void *dev[NLEV];
  double *host[NLEV], *out[NLEV], *ref[NLEV];
  double rg[128] = {0}, rr[64] = {0}, ref_g[128], ref_r[64];
  for (int l = 0; l < NLEV; l++) {
    offload_create_buffer(nr_[l], &buffers[l]);
    assert(!offload_grid_device(buffers[l], sizeof(double) * (size_t)nr_[l]));
    host[l] = buffers[l]->host_buffer;
    out[l] = rarray((size_t)nr_[l]); ref[l] = rarray((size_t)nr_[l]);
    for (int i = 0; i < nr_[l]; i++) host[l][i] = 0.7 + 0.1 * rnd();
    // A different producer stream must finish before PW reads these views.
    offloadMemcpyAsyncHtoD(buffers[l]->device_buffer, host[l], buffers[l]->size, stream_copy);
    offloadStreamSynchronize(stream_copy);
    buffers[l]->grid_device_valid = 1;
    buffers[l]->grid_bytes = buffers[l]->size;
    dev[l] = offload_grid_device(buffers[l], buffers[l]->size);
    assert(dev[l] == buffers[l]->device_buffer);
    assert(!offload_grid_device(buffers[l], buffers[l]->size + 1));
  }
  density_reference(host, ref_g, ref_r);
  for (int pass = 0; pass < 3; pass++) {
    // Mixed device/host input and NULL host pointers are both supported.
    void *inputs[NLEV] = {dev[0], pass == 1 ? NULL : dev[1], dev[2]};
    double *hp[NLEV] = {NULL, pass == 1 ? host[1] : NULL, NULL};
    mock_reset_trace();
    assert(density_device(hp, inputs, rg, rr));
    if (pass > 0) assert(mock_upload_bytes == (pass == 1 ? buffers[1]->size : 0));
    for (size_t i = 0; i < mock_operation_count; i++) {
      if (mock_operations[i].kind == MOCK_REAL_TO_COMPLEX) {
        assert(mock_operations[i].buffer != host[0]);
      }
    }
    pw_gpu_res_ensure_host_c(-1, rg);
    pw_gpu_res_ensure_host_c(-1, rr);
    same(rg, ref_g, 2 * (size_t)ng_[0], "Grid density G");
    same(rr, ref_r, (size_t)nr_[0], "Grid density R");
  }
  double vxc[64], exc, expected[64], vh[64];
  assert(xc_call(rg, rr, vxc, &exc));
  pw_gpu_res_ensure_host_c(RES_VXC, vxc);
  for (int i = 0; i < nr_[0]; i++) { vh[i] = rnd(); expected[i] = 0.125 * vxc[i] + vh[i]; }
  const double scales[NLEV] = {1.0, 8.0, 0.5};
  potential_reference(expected, ref, scales);
  assert(pw_gpu_res_vtot_prepare_c(vh, vxc, (size_t)nr_[0], 0.125));
  assert(pw_gpu_res_vtot_skip_host_c());
  void *alias = NULL;
  for (int i = 0; i < nr_[0]; i++) out[0][i] = -999;
  mock_reset_trace();
  assert(potential_device(vxc, out, &alias));
  assert(alias == res[RES_VXC].dev);
  assert(mock_download_bytes == sizeof(double) * (size_t)(nr_[1] + nr_[2]));
  for (int i = 0; i < nr_[0]; i++) assert(out[0][i] == -999);
  same(out[1], ref[1], (size_t)nr_[1], "coarse potential 1");
  same(out[2], ref[2], (size_t)nr_[2], "coarse potential 2");
  double *owned = buffers[0]->device_buffer;
  assert(offload_grid_bind(buffers[0], alias, (size_t)nr_[0] * sizeof(double)));
  assert(buffers[0]->device_buffer == owned);
  assert(offload_grid_device(buffers[0], buffers[0]->size) == alias);
  assert(!offload_grid_bind(buffers[0], alias, buffers[0]->size + 1));
  // CPU/validation fallback materializes the alias, never the stale owned grid.
  offload_grid_host(buffers[0]);
  same(host[0], expected, (size_t)nr_[0], "borrowed potential fallback");
  assert(!offload_grid_device(buffers[0], buffers[0]->size));
  // Consuming/freeing a borrowed descriptor must not free PW's allocation.
  assert(offload_grid_bind(buffers[0], alias, buffers[0]->size));
  mock_reset_trace();
  offload_grid_prepare(buffers[0], 0);
  assert(mock_download_bytes == 0 && buffers[0]->device_buffer == owned);
  assert(offload_grid_bind(buffers[0], alias, buffers[0]->size));
  offload_free_buffer(buffers[0]); buffers[0] = NULL;
  pw_gpu_res_ensure_host_c(RES_VXC, vxc);
  same(vxc, expected, (size_t)nr_[0], "PW still owns potential");
  // Rejected device output leaves no alias and preserves the fallback input.
  opt_res_pot = false;
  alias = owned;
  assert(!potential_device(vxc, out, &alias) && alias == NULL);
  for (int l = 0; l < NLEV; l++) {
    offload_free_buffer(buffers[l]);
    free(out[l]); free(ref[l]); free(map_[l]); free(gidx_[l]);
  }
  free(g_fine); free(gsq_fine);
  pw_gpu_finalize();
  assert(mock_allocated == 0 && mock_live_streams == 0 && mock_live_events == 0);
  puts("PASS Grid device bridge, mixed inputs, borrowed potential and fallback/ownership");
  return 0;
}
