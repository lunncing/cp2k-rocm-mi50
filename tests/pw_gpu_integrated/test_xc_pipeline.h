/* SPDX-License-Identifier: GPL-2.0-or-later */
#define PW_GPU_ELEMENTAL static inline
#include "pw_gpu_pbe.h"
#include "pw_gpu_xc_ops.h"

static const int xc_dims[3] = {3, 3, 3};
static int xc_map[16];
static double xc_g[24], xc_z[16], xc_rho[27];

static void xc_inputs(void) {
  const int modes[8] = {0, 1, 3, 4, 9, 10, 12, 13};
  for (int i = 0; i < 8; i++) {
    xc_map[i] = modes[i];
    int x = modes[i] % 3, y = modes[i] / 3 % 3, z = modes[i] / 9;
    xc_map[i + 8] = (3 - x) % 3 + 3 * ((3 - y) % 3) + 9 * ((3 - z) % 3);
    xc_g[3*i] = .2*x; xc_g[3*i+1] = .3*y; xc_g[3*i+2] = .4*z;
    xc_z[2*i] = .03*cos(i); xc_z[2*i+1] = i ? .02*sin(i) : 0;
  }
  for (int i = 0; i < 27; i++) xc_rho[i] = .7 + .2*sin(i*.17);
}

static int xc_run(double *out, double *exc, int nmaps, int filter, double cutoff, double range) {
  return pw_gpu_xc_full_pipeline_configured_c(xc_z, xc_rho, out, exc, xc_map, xc_g,
      xc_dims, 8, nmaps, 1.0/27, 1, cutoff, .804, .066725, .2195149727645171,
      .7, .9, .02, range, filter);
}

static void xc_trace(int cached, int nmaps, int filter) {
  assert(mock_download_bytes == 28 * sizeof(double));
  assert(mock_upload_bytes == 43 * sizeof(double) +
         (cached ? 0 : 24 * sizeof(double) + 8 * (size_t)nmaps * sizeof(int)));
  int downloads = 0, ffts = 0, norm = 0, pbe = 0, combine = 0, add = 0;
  bool computing = false;
  for (size_t i = 0; i < mock_operation_count; i++) {
    const mock_operation *op = &mock_operations[i];
    assert(op->stream == stream);
    if (op->kind == MOCK_UPLOAD) assert(!computing);
    else computing = true;
    if (op->kind == MOCK_DOWNLOAD) {
      downloads++;
      assert(i >= mock_operation_count - 2);
    }
    ffts += op->kind == MOCK_FFT;
    norm += op->kind == MOCK_NORM3;
    pbe += op->kind == MOCK_PBE;
    combine += op->kind == MOCK_COMBINE_DIV;
    add += op->kind == MOCK_ADD_VRHO;
  }
  assert(downloads == 2 && ffts == (filter ? 9 : 7));
  assert(norm == 1 && pbe == 1 && combine == 1 && add == 1);
  mock_assert_idle();
  accounting();
}

static void xc_close(const double *a, const double *b, size_t n) {
  for (size_t i = 0; i < n; i++)
    assert(fabs(a[i] - b[i]) <= 1e-12 * (1 + fabs(a[i])));
}

static void xc_rectangular(void) {
  // Unequal axes detect swapped FFT dimensions. The larger grid also needs
  // two reduction passes, putting the final scalar in the FFT scratch before
  // it is copied back to the preserved norm buffer.
  const int shapes[2][3] = {{2, 3, 5}, {3, 5, 19}};
  const int ng = 4;
  for (int shape = 0; shape < 2; shape++) {
    const int *dims = shapes[shape];
    const int n = dims[0]*dims[1]*dims[2];
    const int map[4] = {0, 1, dims[0], dims[0]*dims[1]};
    const double g[12] = {0, 0, 0, .2, 0, 0, 0, .3, 0, 0, 0, .4};
    const double z[8] = {.7, 0, .03, .01, -.02, .03, .01, -.02};
    double *rho = malloc((size_t)n*sizeof(double));
    double *actual = malloc((size_t)n*sizeof(double));
    double *expected = malloc((size_t)n*sizeof(double));
    double *norm = malloc((size_t)n*sizeof(double));
    double *e0 = calloc((size_t)n, sizeof(double));
    double *er = calloc((size_t)n, sizeof(double));
    double *en = calloc((size_t)n, sizeof(double));
    double *grad[3];
    for (int d = 0; d < 3; d++) grad[d] = malloc((size_t)n*sizeof(double));
    const double *inputs[3] = {grad[0], grad[1], grad[2]};
    host_grad3(z, grad, map, g, dims, ng, 1);
    long double energy = 0;
    for (int i = 0; i < n; i++) {
      rho[i] = .7 + .2*sin(.17*i);
      norm[i] = sqrt(grad[0][i]*grad[0][i] + grad[1][i]*grad[1][i] + grad[2][i]*grad[2][i]);
      pw_pbe_lda_elem(rho, norm, e0, er, en, i, 1, 1e-12,
                      .804, .066725, .2195149727645171, .7, .9);
      energy += e0[i];
      for (int d = 0; d < 3; d++) grad[d][i] = -(grad[d][i]*(en[i]/fmax(norm[i], .02)));
    }
    host_div3(inputs, expected, map, g, dims, ng, 1, 1.0/n);
    for (int i = 0; i < n; i++) expected[i] += er[i];
    start(0);
    for (int filter = 0; filter <= 1; filter++) {
      if (filter) host_filter(expected, map, dims, ng, 1, 1.0/n);
      double exc = 0;
      mock_reset_trace();
      assert(pw_gpu_xc_full_pipeline_configured_c(z, rho, actual, &exc, map, g,
          dims, ng, 1, 1.0/n, 1, 1e-12, .804, .066725, .2195149727645171,
          .7, .9, .02, 0, filter));
      xc_close(expected, actual, (size_t)n);
      assert(fabsl(energy - exc) <= 1e-12L*(1 + fabsl(energy)));
      assert(mock_download_bytes == ((size_t)n + 1)*sizeof(double));
      mock_assert_idle();
      accounting();
    }
    finish();
    for (int d = 0; d < 3; d++) free(grad[d]);
    free(rho); free(actual); free(expected); free(norm);
    free(e0); free(er); free(en);
  }
}

static void xc_pipeline(void) {
  xc_inputs();
  double actual[27], expect[27], exc, energy;
  assert(!xc_run(actual, &exc, 2, 1, 1e-12, 0)); // not initialized
  for (int nmaps = 1; nmaps <= 2; nmaps++) for (int filter = 0; filter < 2; filter++)
    for (int smooth = 0; smooth < 2; smooth++) {
      const double cutoff = smooth ? .3 : 1e-12, range = smooth ? 2.0 : 0.0;
      start(0);
      double gradients[3][27], norm[27], e0[27] = {0}, er[27] = {0}, en[27] = {0};
      double *gr[3] = {gradients[0], gradients[1], gradients[2]};
      const double *inputs[3] = {gr[0], gr[1], gr[2]};
      host_grad3(xc_z, gr, xc_map, xc_g, xc_dims, 8, nmaps);
      for (int i = 0; i < 27; i++) {
        norm[i] = sqrt(gr[0][i]*gr[0][i] + gr[1][i]*gr[1][i] + gr[2][i]*gr[2][i]);
        pw_pbe_lda_elem(xc_rho, norm, e0, er, en, i, 1, cutoff,
                        .804, .066725, .2195149727645171, .7, .9);
        for (int d = 0; d < 3; d++) gr[d][i] = -(gr[d][i] * (en[i]/fmax(norm[i], .02)));
        e0[i] = pw_xc_cutoff(e0[i], xc_rho[i], cutoff, range);
      }
      host_div3(inputs, expect, xc_map, xc_g, xc_dims, 8, nmaps, 1.0/27);
      for (int i = 0; i < 27; i++) expect[i] += er[i];
      if (filter) host_filter(expect, xc_map, xc_dims, 8, nmaps, 1.0/27);
      energy = 0;
      for (int i = 0; i < 27; i++) {
        expect[i] = pw_xc_cutoff(expect[i], xc_rho[i], cutoff*range, range);
        energy += e0[i];
      }
      assert(xc_run(actual, &exc, nmaps, filter, cutoff, range));
      xc_close(expect, actual, 27); xc_close(&energy, &exc, 1);
      xc_trace(0, nmaps, filter);
      const size_t allocations = mock_allocations;
      mock_reset_trace();
      assert(xc_run(actual, &exc, nmaps, filter, cutoff, range));
      xc_trace(opt_map_cache, nmaps, filter);
      if (opt_map_cache) assert(mock_allocations == allocations);
      // Input/output alias: all density use precedes the only array download.
      mock_reset_trace();
      double saved[27]; memcpy(saved, xc_rho, sizeof(saved));
      assert(xc_run(xc_rho, &exc, nmaps, filter, cutoff, range));
      xc_close(expect, xc_rho, 27);
      memcpy(xc_rho, saved, sizeof(saved));
      finish();
      assert(xc_dev == NULL && allocated_xc_size == 0);
    }
  xc_rectangular();
}

static void xc_limits(int disabled) {
  xc_inputs();
  const size_t r = 27*sizeof(double), z = sizeof(xc_z), m = sizeof(xc_map), g = sizeof(xc_g);
  const size_t total = 8*r + z + m + g;
  const size_t limits[] = {1, 4*r, 4*r+z, 4*r+z+m, total-1, total};
  for (size_t i = 0; i < sizeof(limits)/sizeof(limits[0]); i++) {
    start(0);
    max_gpu_memory_bytes = mock_limit = limits[i];
    if (disabled) pw_gpu_set_host_fallback(0);
    double out[27], exc = 918;
    for (int j = 0; j < 27; j++) out[j] = 123;
    const int ok = xc_run(out, &exc, 2, 1, 1e-12, 0);
    assert(ok == (limits[i] == total));
    if (!ok) {
      for (int j = 0; j < 27; j++) assert(out[j] == 123);
      assert(exc == 918 && mock_fft_calls == 0 && mock_download_bytes == 0);
    } else assert(current_gpu_allocated_bytes == total);
    mock_assert_idle();
    finish();
  }
  // Retained XC scratch is optional to other operations and the limit setter.
  start(0);
  double out[27], exc;
  assert(xc_run(out, &exc, 2, 0, 1e-12, 0));
  max_gpu_memory_bytes = mock_limit = total;
  assert(pw_gpu_ensure_gbuf_c(z + 1));
  assert(xc_dev == NULL);
  assert(find_grid(1));
  finish();

  // Reclaim the optional transfer buffers before rejecting required XC grids;
  // protect the active grid while evicting other grid cache entries.
  start(0);
  pw_gpu_sfffc_grad3(xc_z, out, out, out, xc_map, xc_g, xc_dims, 8, 2, 1);
  assert(allocated_buffer_size_b != 0);
  grid_cache_entry *other = grid_cache_lookup_or_upload(2, xc_map, 8, 2);
  assert(other && grid_cache_get_g(other, xc_g));
  offloadStreamSynchronize(stream);
  max_gpu_memory_bytes = mock_limit = total;
  assert(xc_run(out, &exc, 2, 0, 1e-12, 0));
  assert(!allocated_buffer_size_b && find_grid(1) && !find_grid(2));
  finish();
}

static void xc_edges(void) {
  xc_inputs(); start(0);
  double out[27], exc = 17;
  for (int i = 0; i < 27; i++) out[i] = 19;
  const int empty[3] = {0, 3, 3}, big[3] = {INT_MAX, INT_MAX, INT_MAX};
  assert(pw_gpu_xc_full_pipeline_c(NULL, NULL, NULL, &exc, NULL, NULL, empty, 0, 1,
                                  1, 1, 1e-12, .804, .066725, .2195, 1, 1));
  assert(exc == 0 && mock_allocated == 0);
  exc = 17;
  assert(!pw_gpu_xc_full_pipeline_c(NULL, NULL, out, &exc, NULL, NULL, big, 1, 1,
                                   1, 1, 1e-12, .804, .066725, .2195, 1, 1));
  assert(exc == 17 && out[0] == 19 && mock_allocated == 0);
  assert(!xc_run(out, &exc, 3, 0, 1e-12, 0));
#pragma omp parallel num_threads(2)
  {
    assert(!xc_run(out, &exc, 2, 0, 1e-12, 0));
  }
  // A real grid with no G coefficients is still a valid LDA evaluation.
  assert(pw_gpu_xc_full_pipeline_c(NULL, xc_rho, out, &exc, NULL, NULL, xc_dims, 0, 1,
                                  1.0/27, 1, 1e-12, .804, .066725, .2195, 1, 1));
  assert(isfinite(exc));
  for (int i = 0; i < 27; i++) assert(isfinite(out[i]));
  // With no retained G coefficients the spherical filter must zero Vxc,
  // while leaving the separately integrated energy unchanged.
  const double unfiltered_energy = exc;
  assert(pw_gpu_xc_full_pipeline_configured_c(NULL, xc_rho, out, &exc, NULL,
      NULL, xc_dims, 0, 1, 1.0/27, 1, 1e-12, .804, .066725, .2195, 1, 1,
      1e-12, 0, 1));
  assert(exc == unfiltered_energy);
  for (int i = 0; i < 27; i++) assert(out[i] == 0.0);
  finish();
  start(0);
  opt_fused_xc = false;
  exc = 17;
  assert(!xc_run(out, &exc, 2, 1, 1e-12, 0));
  assert(exc == 17 && mock_allocated == 0);
  finish();
}
