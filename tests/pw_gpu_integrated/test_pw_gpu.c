/* SPDX-License-Identifier: GPL-2.0-or-later */
#define _POSIX_C_SOURCE 200809L
// Compile the actual implementation to exercise its internal allocation paths.
#include "pw_gpu_internal.c"
#include "mock_offload.h"

static void accounting(void) {
  size_t allocated, peak;
  pw_gpu_get_allocated_bytes(&allocated, &peak);
  assert(allocated == mock_allocated);
  assert(peak == mock_peak);
  assert(max_gpu_memory_bytes == 0 || allocated <= max_gpu_memory_bytes);
}

static void start(const int mb) {
  assert(mock_allocated == 0 && mock_live_plans == 0);
  assert(mock_live_streams == 0 && mock_live_events == 0);
  mock_peak = mock_allocations = 0;
  mock_uploads = mock_fft_calls = 0;
  pw_gpu_set_memory_limit(mb);
  mock_limit = max_gpu_memory_bytes;
  pw_gpu_init();
  assert(mock_live_streams == 2 && mock_live_events == 4);
  mock_reset_trace();
  accounting();
}

static void finish(void) {
  accounting();
  pw_gpu_finalize();
  accounting();
  assert(mock_allocated == 0 && mock_live_plans == 0);
  assert(mock_live_streams == 0 && mock_live_events == 0);
  assert(buffer_dev_1 == NULL && buffer_dev_2 == NULL);
  assert(buffer_dev_1_b == NULL && buffer_dev_2_b == NULL && allocated_buffer_size_b == 0);
  assert(gbuf_dev == NULL && ghatmap_dev == NULL);
  pw_gpu_finalize(); // idempotent
}

static grid_cache_entry *find_grid(const int id) {
  for (int i = 0; i < PW_GPU_GRID_CACHE_SIZE; i++) {
    if (grid_cache[i].grid_id == id) return &grid_cache[i];
  }
  return NULL;
}

static void limits(void) {
  start(1);
  assert(mock_allocations == 0); // Initialization is lazy.
  assert(pw_gpu_ensure_buffer_c(PW_GPU_MIB / 8, PW_GPU_MIB / 8, false));
  assert(pw_gpu_ensure_gbuf_c(PW_GPU_MIB / 8));
  assert(current_gpu_allocated_bytes == PW_GPU_MIB / 2);
  // Net growth, not the full replacement, is charged. Final usage is exactly 1 MiB.
  assert(pw_gpu_ensure_buffer_c(3 * PW_GPU_MIB / 8, PW_GPU_MIB / 8, false));
  assert(current_gpu_allocated_bytes == PW_GPU_MIB);
  accounting();
  void *old = buffer_dev_1;
  const size_t calls = mock_allocations;
  assert(!pw_gpu_ensure_buffer_c(3 * PW_GPU_MIB / 8 + 1, 0, false));
  assert(!pw_gpu_ensure_gbuf_c(PW_GPU_MIB / 8 + 1));
  assert(!pw_gpu_ensure_buffer_c(0, PW_GPU_MIB / 8 + 1, false));
  assert(!pw_gpu_ensure_buffer_c(SIZE_MAX, SIZE_MAX, false));
  assert(!check_and_reclaim_memory(SIZE_MAX));
  assert(mock_allocations == calls && buffer_dev_1 == old);
  accounting();
  pw_gpu_set_memory_limit(0);
  mock_limit = 0;
  assert(pw_gpu_ensure_buffer_c(PW_GPU_MIB, 0, false));
  accounting();
  // Lowering a live budget discards idle scratch safely; peak is retained.
  pw_gpu_set_memory_limit(1);
  mock_limit = PW_GPU_MIB;
  assert(current_gpu_allocated_bytes == 0);
  assert(pw_gpu_ensure_buffer_c(PW_GPU_MIB / 2, 0, false));
  assert(current_gpu_allocated_bytes == PW_GPU_MIB);
  finish();
  start(1); // Reinitialization resets the peak, but preserves configuration.
  assert(peak_gpu_allocated_bytes == 0);
  finish();
}

static void eviction(void) {
  static int maps[8192];
  static double g[12288], changed_g[12288];
  start(1);
  for (int id = 1; id <= 8; id++) {
    grid_cache_entry *e = grid_cache_lookup_or_upload(id, maps, 4096, 2);
    assert(e && grid_cache_get_g(e, g));
  }
  assert(current_gpu_allocated_bytes == PW_GPU_MIB);
  const int uploads = mock_uploads;
  grid_cache_entry *e = grid_cache_lookup_or_upload(3, maps, 4096, 2);
  assert(e && grid_cache_get_g(e, g));
  assert(mock_uploads == uploads); // Both maps and g-vectors remain resident.
  assert(pw_gpu_ensure_buffer_c(65536, 0, false));
  assert(!find_grid(1) && find_grid(2) && find_grid(3));
  assert(pw_gpu_ensure_gbuf_c(131072));
  assert(!find_grid(2) && find_grid(3));
  assert(current_gpu_allocated_bytes == PW_GPU_MIB);
  accounting();
  assert(grid_cache_get_g(e, changed_g));
  assert(mock_uploads == uploads + 1 && e->g_host == changed_g);
  pw_gpu_grid_invalidate(3);
  assert(!find_grid(3));
  assert(current_gpu_allocated_bytes == PW_GPU_MIB - 131072);
  pw_gpu_grid_invalidate(3); // Repeated invalidation does not subtract twice.
  finish();

  // Pinning protects the oldest map while its g-vector allocation evicts a
  // newer entry; without it e would become a dangling/empty cache entry.
  start(1);
  e = grid_cache_lookup_or_upload(1, maps, 4096, 2);
  grid_cache_entry *other = grid_cache_lookup_or_upload(2, maps, 4096, 2);
  assert(other && grid_cache_get_g(other, g));
  assert(pw_gpu_ensure_gbuf_c(PW_GPU_MIB - 32768 - 131072));
  assert(grid_cache_get_g(e, g));
  assert(find_grid(1) == e && e->map_dev && e->g_dev && !find_grid(2));
  accounting();
  finish();

  // Capacity eviction still obeys LRU with an unlimited memory budget.
  start(0);
  for (int id = 1; id <= 8; id++)
    assert(grid_cache_lookup_or_upload(id, maps, 4096, 2));
  assert(grid_cache_lookup_or_upload(1, maps, 4096, 2));
  assert(grid_cache_lookup_or_upload(9, maps, 4096, 2));
  assert(find_grid(1) && !find_grid(2) && find_grid(9));
  e = grid_cache_lookup_or_upload(1, maps + 1, 2048, 1);
  assert(e && e->ngpts == 2048 && e->nmaps == 1 && e->map_host == maps + 1);
  accounting();
  finish();
}

static void close_arrays(const double *a, const double *b, const size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (!(fabs(a[i] - b[i]) <= 2e-11 * (1 + fabs(a[i])))) {
      fprintf(stderr, "Numerical mismatch at %zu: %.17g != %.17g\n", i, a[i], b[i]);
      abort();
    }
  }
}

// Cover every public GPU transformation, full/half-space maps, both directions,
// the distributed-grid transposes, and in-place divergence/filter/complex FFT.
static size_t transforms(double *out, const int nmaps) {
  const int npts[] = {3, 2, 4}, ng = 8;
  int map[16];
  double r[3][24], z[48], g[24], alias[48];
  for (int i = 0; i < 8; i++) { map[i] = i; map[i + 8] = 16 + i; }
  for (int i = 0; i < 24; i++) {
    for (int d = 0; d < 3; d++) r[d][i] = sin(0.17 * (i + 1) * (d + 1));
    z[2 * i] = cos(0.11 * i); z[2 * i + 1] = sin(0.23 * i);
    g[i] = 0.1 * (i % 7 - 3);
  }
  size_t k = 0;
  pw_gpu_cfffg_cached(r[0], out + k, map, npts, ng, nmaps, 1.0 / 24, 1); k += 16;
  pw_gpu_sfffc_cached(z, out + k, map, npts, ng, nmaps, 0.7, 1); k += 24;
  pw_gpu_cfffg(r[1], out + k, map, npts, ng, 0.2); k += 16;
  pw_gpu_sfffc(z, out + k, map, npts, ng, nmaps, 0.3); k += 24;
  pw_gpu_sfffc_grad3(z, out + k, out + k + 24, out + k + 48,
                     map, g, npts, ng, nmaps, 2); k += 72;
  pw_gpu_cfffg_div3(r[0], r[1], r[2], out + k, map, g, npts, ng, nmaps, 1.0 / 24, 2); k += 24;
  memcpy(alias, r[0], sizeof(r[0]));
  pw_gpu_cfffg_div3(alias, r[1], r[2], alias, map, g, npts, ng, nmaps, 1.0 / 24, 2);
  memcpy(out + k, alias, sizeof(r[0])); k += 24;
  memcpy(out + k, r[0], sizeof(r[0]));
  pw_gpu_filter_r3d(out + k, map, npts, ng, nmaps, 1.0 / 24, 3); k += 24;
  pw_gpu_cff(r[0], out + k, npts); k += 48;
  pw_gpu_ffc(z, out + k, npts); k += 24;
  pw_gpu_cf(r[0], out + k, npts); k += 48;
  pw_gpu_fc(z, out + k, npts); k += 24;
  pw_gpu_f(z, out + k, 1, 6, 4); k += 48;
  pw_gpu_f(z, out + k, -1, 6, 4); k += 48;
  memcpy(alias, z, sizeof(z));
  pw_gpu_f(alias, alias, 1, 6, 4);
  memcpy(out + k, alias, sizeof(alias)); k += 48;
  pw_gpu_fg(z, out + k, map, npts, 8, ng, 0.3); k += 16;
  pw_gpu_sf(z, out + k, map, npts, 8, ng, nmaps, 0.3); k += 48;
  accounting();
  return k;
}

static void fallback(void) {
  double expected[1024], actual[1024];
  for (int nmaps = 1; nmaps <= 2; nmaps++) {
    start(0);
    assert(pw_gpu_fused_xc_enabled());
    size_t n = transforms(expected, nmaps);
    assert(mock_fft_calls > 0);
    finish();
    start(1);
    assert(pw_gpu_ensure_gbuf_c(PW_GPU_MIB));
    assert(transforms(actual, nmaps) == n);
    // No GPU FFT, copy or allocation may slip through after a failed reservation.
    assert(mock_fft_calls == 0 && mock_uploads == 0 && mock_allocations == 1);
    close_arrays(expected, actual, n);
    finish();
  }
}

static void late_fallback(void) {
  const int npts[] = {2, 2, 2}, ng = 2, nmaps = 2, map[] = {1, 2, 6, 5};
  const double z[] = {0.3, 0.2, 0.4, -0.1}, g[] = {1, 2, 3, -2, 1, 0};
  double expected[24], actual[24], data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  start(0);
  pw_gpu_sfffc_grad3(z, expected, expected + 8, expected + 16, map, g, npts, ng, nmaps, 1);
  finish();
  for (int stage = 0; stage < 3; stage++) {
    start(1);
    // Fail at gbuf, map, then g-vectors, after earlier reservations succeeded.
    const size_t gbytes = 2 * sizeof(double) * ng;
    const size_t mbytes = sizeof(int) * ng * nmaps;
    size_t available = stage == 0 ? 0 : stage == 1 ? gbytes : gbytes + mbytes;
    assert(pw_gpu_ensure_buffer_c((PW_GPU_MIB - available) / 2, 0, false));
    pw_gpu_sfffc_grad3(z, actual, actual + 8, actual + 16, map, g, npts, ng, nmaps, 1);
    assert(mock_fft_calls == 0);
    close_arrays(expected, actual, 24);
    if (stage == 2) assert(find_grid(1) && find_grid(1)->map_dev && !find_grid(1)->g_dev);
    finish();
  }
  // Ordinary cached and legacy map failures also return numerical results.
  start(0);
  pw_gpu_cfffg(data, expected, map, npts, ng, 0.125);
  finish();
  start(1);
  assert(pw_gpu_ensure_buffer_c(PW_GPU_MIB / 2, 0, false));
  pw_gpu_cfffg(data, actual, map, npts, ng, 0.125);
  close_arrays(expected, actual, 4);
  pw_gpu_cfffg_cached(data, actual, map, npts, ng, nmaps, 0.125, 1);
  close_arrays(expected, actual, 4);
  assert(mock_fft_calls == 0);
  finish();
}

static bool ordered(const mock_operation *a, const mock_operation *b) {
  for (int s = 1; s <= 2; s++) if (a->clock[s] > b->clock[s]) return false;
  return true;
}

static const mock_operation *operation(enum mock_operation_kind kind, int index) {
  for (size_t i = 0; i < mock_operation_count; i++) {
    if (mock_operations[i].kind == kind && index-- == 0) return &mock_operations[i];
  }
  assert(!"missing operation");
  return NULL;
}

static void concurrent(const mock_operation *a, const mock_operation *b) {
  assert(a->stream != b->stream && !ordered(a, b) && !ordered(b, a));
}

static void pipeline_trace(const bool gradient, const bool overlap) {
  const mock_operation *copies[3];
  int count = 0;
  mock_assert_idle(); // Both streams must finish before returning host results.
  for (size_t i = 0; i < mock_operation_count; i++) {
    const mock_operation *op = &mock_operations[i];
    if (!overlap) assert(op->stream == stream);
    if ((gradient && op->kind == MOCK_DOWNLOAD) ||
        (!gradient && op->kind == MOCK_UPLOAD &&
         (op->buffer == buffer_dev_1 || op->buffer == buffer_dev_1_b))) {
      assert(count < 3);
      copies[count++] = op;
    }
  }
  assert(count == 3);
  assert(mock_event_records == (overlap ? 6 : 0));
  assert(mock_event_waits == (overlap ? 4 : 0));
  for (int d = 0; d < 3; d++) {
    const mock_operation *fft = operation(MOCK_FFT, d);
    const mock_operation *deriv = operation(gradient ? MOCK_SCATTER_DERIV : MOCK_GATHER_DERIV, d);
    assert(fft->stream == stream && deriv->stream == stream && deriv->direction == d);
    assert(fft->direction == (gradient ? OFFLOAD_FFT_INVERSE : OFFLOAD_FFT_FORWARD));
    assert(fft->buffer == (overlap && d == 1 ? buffer_dev_2_b : buffer_dev_2));
    assert(copies[d]->buffer == (overlap && d == 1 ? buffer_dev_1_b : buffer_dev_1));
    assert(copies[d]->stream == (overlap ? stream_copy : stream));
    if (gradient) {
      assert(ordered(deriv, fft) && ordered(fft, copies[d]));
      if (overlap && d > 0) {
        concurrent(operation(MOCK_MEMSET, d), copies[d - 1]);
        concurrent(deriv, copies[d - 1]);
        concurrent(fft, copies[d - 1]);
      }
    } else {
      assert(ordered(copies[d], fft) && ordered(fft, deriv));
      assert(ordered(copies[d], operation(MOCK_DOWNLOAD, 0))); // Includes aliased inputs.
      if (d > 0) assert(ordered(operation(MOCK_GATHER_DERIV, d - 1), deriv));
      if (overlap && d < 2) {
        concurrent(fft, copies[d + 1]);
        concurrent(deriv, copies[d + 1]);
      }
    }
  }
  if (gradient) {
    assert(ordered(copies[0], operation(MOCK_MEMSET, 2)));
  } else {
    assert(ordered(operation(MOCK_GATHER_DERIV, 0), copies[2]));
    assert(operation(MOCK_FFT, 3)->direction == OFFLOAD_FFT_INVERSE);
  }
}

static void dual_stream(void) {
  const int npts[] = {3, 2, 4}, ng = 8;
  int map[16];
  double z[16], g[24], input[3][24], alias[3][24];
  double expected_grad[3][24], expected_div[24], grad[3][24], div[24];
  for (int i = 0; i < 8; i++) { map[i] = i; map[i + 8] = 16 + i; }
  for (int i = 0; i < 16; i++) z[i] = sin(0.23 * i);
  for (int i = 0; i < 24; i++) {
    g[i] = i % 5 == 0 ? -0.0 : 0.1 * (i % 7 - 3);
    for (int d = 0; d < 3; d++) input[d][i] = cos(0.17 * (d + 1) * i);
  }
  z[0] = input[0][0] = -0.0;
  for (int nmaps = 1; nmaps <= 2; nmaps++) {
    const size_t buffer_size = 2 * sizeof(double) * 24;
    const size_t mandatory = 2 * buffer_size + 5 * sizeof(double) * ng + sizeof(int) * ng * nmaps;
    const size_t extra = buffer_size + buffer_size / 2;
    // Exact serial budget, one byte short of dual, exact dual budget, unlimited.
    const size_t budgets[] = {mandatory, mandatory + extra - 1, mandatory + extra, 0};
    for (int mode = 0; mode < 4; mode++) {
      start(0);
      // Byte-level test limits exercise boundaries hidden by the MiB public setter.
      max_gpu_memory_bytes = mock_limit = budgets[mode];
      pw_gpu_set_host_fallback(mode % 2);
      const bool overlap = mode >= 2;
      for (int repeat = 0; repeat < 2; repeat++) {
        mock_reset_trace();
        pw_gpu_sfffc_grad3(z, grad[0], grad[1], grad[2], map, g, npts, ng, nmaps, 1);
        pipeline_trace(true, overlap);
        if (mode == 0 && repeat == 0) memcpy(expected_grad, grad, sizeof(grad));
        assert(memcmp(expected_grad, grad, sizeof(grad)) == 0);
        mock_reset_trace();
        pw_gpu_cfffg_div3(input[0], input[1], input[2], div, map, g, npts, ng, nmaps, 1.0 / 24, 1);
        pipeline_trace(false, overlap);
        if (mode == 0 && repeat == 0) memcpy(expected_div, div, sizeof(div));
        assert(memcmp(expected_div, div, sizeof(div)) == 0);
        for (int d = 0; d < 3; d++) {
          memcpy(alias, input, sizeof(alias));
          mock_reset_trace();
          pw_gpu_cfffg_div3(alias[0], alias[1], alias[2], alias[d], map, g, npts, ng, nmaps, 1.0 / 24, 1);
          pipeline_trace(false, overlap);
          assert(memcmp(expected_div, alias[d], sizeof(div)) == 0);
        }
        assert(mock_allocations == (overlap ? 7 : 5)); // No partial optional allocation or churn.
        assert(current_gpu_allocated_bytes == mandatory + (overlap ? extra : 0));
        assert(!fallback_notice_printed);
        accounting();
      }
      finish();
    }
  }
}

static void dual_buffer_memory(void) {
  double host[64] = {0};
  const int map[] = {0, 1, 2, 3};
  const double g[] = {1, 2, 3, 4, 5, 6};
  start(0);
  pw_gpu_set_host_fallback(0);
  assert(pw_gpu_ensure_buffer_c(256, 0, true));
  assert(current_gpu_allocated_bytes == 896 && allocated_buffer_size_b == 256);
  double *old_b = buffer_dev_1_b;
  const size_t allocations_before = mock_allocations;
  assert(pw_gpu_ensure_buffer_c(128, 0, true));
  assert(buffer_dev_1_b == old_b && mock_allocations == allocations_before);
  // Growth must drain pending work on both streams before freeing either buffer.
  offloadMemcpyAsyncHtoD(buffer_dev_1_b, host, 128, stream_copy);
  offloadMemsetAsync(buffer_dev_2_b, 0, 256, stream);
  assert(pw_gpu_ensure_buffer_c(512, 0, true));
  assert(current_gpu_allocated_bytes == 1792 && allocated_buffer_size_b == 512);
  grid_cache_entry *e = grid_cache_lookup_or_upload(1, map, 2, 2);
  assert(e && grid_cache_get_g(e, g));
  max_gpu_memory_bytes = mock_limit = current_gpu_allocated_bytes;
  // Required scratch reclaims the optional pair before evicting a cached grid.
  assert(pw_gpu_ensure_gbuf_c(32));
  assert(allocated_buffer_size_b == 0 && find_grid(1) == e && e->g_dev);
  assert(current_gpu_allocated_bytes == 2 * 512 + 64 + 32);
  finish();

  start(0);
  assert(pw_gpu_ensure_buffer_c(128, 0, false));
  assert(pw_gpu_ensure_gbuf_c(32));
  e = grid_cache_lookup_or_upload(1, map, 2, 2);
  assert(e && grid_cache_get_g(e, g));
  for (int id = 2; id <= 3; id++) {
    grid_cache_entry *other = grid_cache_lookup_or_upload(id, map, 2, 2);
    assert(other && grid_cache_get_g(other, g));
  }
  // The active (oldest) grid survives while the next LRU entry is reclaimed.
  max_gpu_memory_bytes = mock_limit = 608;
  protected_grid_entry = e;
  assert(pw_gpu_ensure_buffer_c(128, 0, true));
  protected_grid_entry = NULL;
  assert(find_grid(1) == e && !find_grid(2) && find_grid(3));
  assert(e->map_dev && e->g_dev && current_gpu_allocated_bytes == 608);
  release_double_buffers();
  max_gpu_memory_bytes = mock_limit = current_gpu_allocated_bytes;
  protected_grid_entry = e;
  assert(pw_gpu_ensure_buffer_c(128, 0, true)); // Not enough, even after LRU eviction.
  protected_grid_entry = NULL;
  assert(find_grid(1) == e && e->g_dev && !find_grid(3));
  assert(!buffer_dev_1_b && !buffer_dev_2_b && !fallback_notice_printed);
  finish();

  start(0);
  assert(pw_gpu_ensure_buffer_c(PW_GPU_MIB / 4, 0, true));
  assert(pw_gpu_ensure_gbuf_c(PW_GPU_MIB / 2));
  double *old_a = buffer_dev_1;
  pw_gpu_set_memory_limit(1);
  mock_limit = PW_GPU_MIB;
  assert(current_gpu_allocated_bytes == PW_GPU_MIB && !buffer_dev_1_b && !buffer_dev_2_b);
  assert(buffer_dev_1 == old_a && gbuf_dev); // Keep required scratch after lowering the limit.
  assert(pw_gpu_ensure_buffer_c(PW_GPU_MIB / 4, 0, true));
  assert(!buffer_dev_1_b && !buffer_dev_2_b && !fallback_notice_printed);
  finish();
}

static void empty_grids(void) {
  const int empty[] = {0, 2, 2}, npts[] = {2, 2, 2};
  double values[24];
  for (int i = 0; i < 24; i++) values[i] = 9;
  start(1);
  pw_gpu_sfffc_grad3(NULL, values, values + 8, values + 16, NULL, NULL, npts, 0, 1, 1);
  for (int i = 0; i < 24; i++) assert(values[i] == 0);
  pw_gpu_filter_r3d(values, NULL, empty, 0, 1, 1, 1);
  pw_gpu_cfffg_div3(NULL, NULL, NULL, values, NULL, NULL, npts, 0, 1, 1, 1);
  pw_gpu_cfffg(NULL, values, NULL, empty, 0, 1);
  pw_gpu_sfffc(NULL, values, NULL, empty, 0, 1, 1);
  pw_gpu_f(NULL, NULL, 1, 0, 1);
  assert(mock_allocations == 0 && mock_fft_calls == 0 && !fallback_notice_printed);
  finish();
}

static void environment(const int argc, char **argv) {
  assert(argc == 6);
  size_t allocated, peak;
  // Environment detection must work on the first query, before initialization,
  // or on initialization itself, without any Fortran input wiring.
  if (atoi(argv[4])) pw_gpu_init();
  else pw_gpu_get_allocated_bytes(&allocated, &peak);
  assert(max_gpu_memory_bytes == (size_t)strtoull(argv[2], NULL, 10) * PW_GPU_MIB);
  assert(host_fallback_enabled == atoi(argv[3]));
  if (atoi(argv[5])) {
    const size_t limit = max_gpu_memory_bytes;
    const int enabled = host_fallback_enabled;
    pw_gpu_set_memory_limit(17);
    pw_gpu_set_host_fallback(!enabled);
    if (memory_limit_from_environment) assert(max_gpu_memory_bytes == limit);
    else assert(max_gpu_memory_bytes == 17 * PW_GPU_MIB);
    if (host_fallback_from_environment) assert(host_fallback_enabled == enabled);
    else assert(host_fallback_enabled == !enabled);
  }
  pw_gpu_finalize();
}

static void fft_sizes(void) {
  // Prime and mixed-radix sizes; compare independently computed complex FFTs.
  double in[256], expected[256], actual[256];
  for (int i = 0; i < 256; i++) in[i] = sin(i * 0.19);
  const int sizes[] = {1, 2, 3, 5, 7, 8, 12, 15, 25};
  start(0);
  for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
    for (int sign = -1; sign <= 1; sign += 2) {
      pw_gpu_f(in, expected, -sign, sizes[i], 3);
      host_fft_1d(sign, sizes[i], 3, in, actual);
      close_arrays(expected, actual, 6 * (size_t)sizes[i]);
    }
  }
  finish();
}

static void environment_once(void) {
  size_t allocated, peak;
  pw_gpu_get_allocated_bytes(&allocated, &peak);
  assert(max_gpu_memory_bytes == 0 && host_fallback_enabled);
#ifdef _WIN32
  assert(_putenv_s("CP2K_GPU_MEM_LIMIT_MB", "31") == 0);
  assert(_putenv_s("CP2K_GPU_HOST_FALLBACK", "FALSE") == 0);
#else
  assert(setenv("CP2K_GPU_MEM_LIMIT_MB", "31", 1) == 0);
  assert(setenv("CP2K_GPU_HOST_FALLBACK", "FALSE", 1) == 0);
#endif
  pw_gpu_init();
  assert(max_gpu_memory_bytes == 0 && host_fallback_enabled);
  finish();
}

static void environment_limit(void) {
  // Exercise an actual allocation rejection with only automatic C detection.
  // The runner gives rank 1 a 1 MiB budget and the global setting 3 MiB.
  pw_gpu_init();
  mock_limit = PW_GPU_MIB;
  assert(max_gpu_memory_bytes == PW_GPU_MIB);
  assert(pw_gpu_ensure_buffer_c(PW_GPU_MIB / 2, 0, false));
  assert(!pw_gpu_ensure_buffer_c(PW_GPU_MIB / 2 + 1, 0, false));
  assert(mock_allocations == 2);
  finish();
}

int main(int argc, char **argv) {
  assert(argc >= 2);
  if (!strcmp(argv[1], "limits")) limits();
  else if (!strcmp(argv[1], "eviction")) eviction();
  else if (!strcmp(argv[1], "fallback")) fallback();
  else if (!strcmp(argv[1], "late_fallback")) late_fallback();
  else if (!strcmp(argv[1], "dual_stream")) dual_stream();
  else if (!strcmp(argv[1], "dual_buffer_memory")) dual_buffer_memory();
  else if (!strcmp(argv[1], "empty")) empty_grids();
  else if (!strcmp(argv[1], "fft_sizes")) fft_sizes();
  else if (!strcmp(argv[1], "environment")) environment(argc, argv);
  else if (!strcmp(argv[1], "environment_once")) environment_once();
  else if (!strcmp(argv[1], "environment_limit")) environment_limit();
  else if (!strcmp(argv[1], "disabled")) {
    start(1);
    pw_gpu_set_host_fallback(0);
    pw_gpu_ensure_buffer_c(PW_GPU_MIB, 0, false);
    assert(!"disabled fallback must fail before launching GPU work");
  } else assert(!"unknown test");
  puts("PASS");
  return 0;
}
