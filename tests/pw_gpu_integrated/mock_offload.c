/* SPDX-License-Identifier: GPL-2.0-or-later */
#include "mock_offload.h"
#include "pw_gpu_kernels.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PW_GPU_ELEMENTAL static inline
#include "pw_gpu_pbe.h"
#include "pw_gpu_xc_ops.h"

size_t mock_allocated, mock_peak, mock_limit, mock_allocations;
size_t mock_upload_bytes, mock_download_bytes;
int mock_fft_calls, mock_uploads, mock_live_plans;
int mock_live_streams, mock_live_events, mock_event_records, mock_event_waits;
int mock_grid_calls, mock_hartree_calls, mock_pbe_calls;
mock_operation mock_operations[8192];
size_t mock_operation_count;
struct mock_fft_plan {
  int rank, dims[3], batch, istride, idist, ostride, odist;
  offloadStream_t stream;
};
static struct {
  void *ptr;
  size_t size;
  size_t last_use[3];
} allocations[64];

// Execute arithmetic immediately, but track the asynchronous happens-before
// graph with vector clocks. Host call order alone must not hide device races.
static struct {
  bool live;
  size_t clock[3];
} streams[3];
static struct {
  bool live, recorded;
  size_t clock[3];
} events[5];
static size_t completed[3];
static offloadStream_t active_stream;

static void merge_clock(size_t *dst, const size_t *src) {
  for (int i = 1; i <= 2; i++) {
    if (dst[i] < src[i]) dst[i] = src[i];
  }
}

static void advance(offloadStream_t stream) {
  assert(stream >= 1 && stream <= 2 && streams[stream].live);
  merge_clock(streams[stream].clock, completed);
  streams[stream].clock[stream]++;
}

static void begin_operation(enum mock_operation_kind kind, offloadStream_t stream,
                            const void *buffer, int direction) {
  advance(stream);
  active_stream = stream;
  assert(mock_operation_count < sizeof(mock_operations) / sizeof(mock_operations[0]));
  mock_operation *op = &mock_operations[mock_operation_count++];
  *op = (mock_operation){kind, stream, buffer, direction, {0}};
  memcpy(op->clock, streams[stream].clock, sizeof(op->clock));
}

void mock_assert_idle(void) {
  for (int i = 1; i <= 2; i++) {
    assert(!streams[i].live || completed[i] == streams[i].clock[i]);
  }
}

void mock_reset_trace(void) {
  mock_assert_idle();
  mock_operation_count = 0;
  mock_upload_bytes = mock_download_bytes = 0;
  mock_event_records = mock_event_waits = 0;
}

static void touch(const void *ptr, size_t bytes) {
  for (size_t i = 0; i < 64; i++) {
    const uintptr_t base = (uintptr_t)allocations[i].ptr, p = (uintptr_t)ptr;
    if (base != 0 && p >= base && p - base < allocations[i].size) {
      assert(bytes <= allocations[i].size - (p - base));
      for (int s = 1; s <= 2; s++) {
        assert(allocations[i].last_use[s] <= streams[active_stream].clock[s] &&
               "device access without a stream dependency");
      }
      allocations[i].last_use[active_stream] = streams[active_stream].clock[active_stream];
      return;
    }
  }
  assert(!"invalid or freed device pointer");
}

void offload_activate_chosen_device(void) {}
void offloadMalloc(void **ptr, size_t bytes) {
  assert(bytes > 0);
  assert(mock_limit == 0 || mock_allocated + bytes <= mock_limit);
  for (size_t i = 0; i < 64; i++) {
    if (allocations[i].ptr == NULL) {
      *ptr = malloc(bytes);
      assert(*ptr != NULL);
      allocations[i].ptr = *ptr;
      allocations[i].size = bytes;
      mock_allocated += bytes;
      mock_allocations++;
      if (mock_allocated > mock_peak) mock_peak = mock_allocated;
      return;
    }
  }
  assert(!"too many mock allocations");
}

void offloadFree(void *ptr) {
  assert(ptr != NULL);
  for (size_t i = 0; i < 64; i++) {
    if (allocations[i].ptr == ptr) {
      for (int s = 1; s <= 2; s++) {
        assert(allocations[i].last_use[s] <= completed[s] &&
               "free before asynchronous use completes");
      }
      mock_allocated -= allocations[i].size;
      free(ptr);
      memset(&allocations[i], 0, sizeof(allocations[i]));
      return;
    }
  }
  assert(!"double free");
}

void offloadStreamCreate(offloadStream_t *stream) {
  for (int i = 1; i <= 2; i++) {
    if (!streams[i].live) {
      memset(&streams[i], 0, sizeof(streams[i]));
      streams[i].live = true;
      completed[i] = 0;
      *stream = i;
      mock_live_streams++;
      return;
    }
  }
  assert(!"too many streams");
}
void offloadStreamDestroy(offloadStream_t stream) {
  assert(stream >= 1 && stream <= 2 && streams[stream].live);
  assert(completed[stream] == streams[stream].clock[stream]);
  streams[stream].live = false;
  mock_live_streams--;
}
void offloadStreamSynchronize(offloadStream_t stream) {
  assert(stream >= 1 && stream <= 2 && streams[stream].live);
  merge_clock(completed, streams[stream].clock);
}
void offloadDeviceSynchronize(void) {
  for (int i = 1; i <= 2; i++)
    if (streams[i].live) merge_clock(completed, streams[i].clock);
}
void offloadEventCreate(offloadEvent_t *event) {
  for (int i = 1; i <= 4; i++) {
    if (!events[i].live) {
      memset(&events[i], 0, sizeof(events[i]));
      events[i].live = true;
      *event = i;
      mock_live_events++;
      return;
    }
  }
  assert(!"too many events");
}
void offloadEventDestroy(offloadEvent_t event) {
  assert(event >= 1 && event <= 4 && events[event].live);
  for (int s = 1; s <= 2; s++) assert(events[event].clock[s] <= completed[s]);
  events[event].live = false;
  mock_live_events--;
}
void offloadEventRecord(offloadEvent_t event, offloadStream_t stream) {
  assert(event >= 1 && event <= 4 && events[event].live);
  advance(stream);
  memcpy(events[event].clock, streams[stream].clock, sizeof(events[event].clock));
  events[event].recorded = true;
  mock_event_records++;
}
void offloadStreamWaitEvent(offloadStream_t stream, offloadEvent_t event) {
  assert(event >= 1 && event <= 4 && events[event].live && events[event].recorded);
  advance(stream);
  // Capture this recording, not any subsequent reuse of the event.
  merge_clock(streams[stream].clock, events[event].clock);
  mock_event_waits++;
}
void offloadMemcpyAsyncHtoD(void *dst, const void *src, size_t bytes,
                           offloadStream_t stream) {
  begin_operation(MOCK_UPLOAD, stream, dst, 0);
  touch(dst, bytes);
  memcpy(dst, src, bytes);
  mock_uploads++;
  mock_upload_bytes += bytes;
}
void offloadMemcpyAsyncDtoH(void *dst, const void *src, size_t bytes,
                           offloadStream_t stream) {
  begin_operation(MOCK_DOWNLOAD, stream, src, 0);
  touch(src, bytes);
  memcpy(dst, src, bytes);
  mock_download_bytes += bytes;
}
void offloadMemsetAsync(void *dst, int value, size_t bytes,
                        offloadStream_t stream) {
  begin_operation(MOCK_MEMSET, stream, dst, 0);
  touch(dst, bytes);
  memset(dst, value, bytes);
}

void offload_fftPlanMany(offload_fftHandle *plan, int rank, int *dims,
                         int *inembed, int istride, int idist, int *onembed,
                         int ostride, int odist, int type, int batch) {
  assert(rank == 1 && inembed[0] == dims[0] && onembed[0] == dims[0]);
  assert(type == OFFLOAD_FFT_Z2Z);
  *plan = malloc(sizeof(**plan));
  assert(*plan != NULL);
  **plan = (struct mock_fft_plan){rank, {dims[0], 1, 1}, batch,
                                 istride, idist, ostride, odist, 0};
  mock_live_plans++;
}
void offload_fftPlan3d(offload_fftHandle *plan, int nx, int ny, int nz, int type) {
  assert(type == OFFLOAD_FFT_Z2Z);
  *plan = malloc(sizeof(**plan));
  assert(*plan != NULL);
  **plan = (struct mock_fft_plan){3, {nx, ny, nz}, 1, 1, 1, 1, 1, 0};
  mock_live_plans++;
}
void offload_fftSetStream(offload_fftHandle plan, offloadStream_t stream) {
  assert(plan->rank > 0 && stream == 1);
  plan->stream = stream;
}
void offload_fftDestroy(offload_fftHandle plan) {
  assert(plan->rank > 0 && mock_live_plans > 0);
  free(plan);
  mock_live_plans--;
}

// Independent direct DFT, deliberately not the production host FFT algorithm.
void offload_fftExecZ2Z(offload_fftHandle plan, const double *in, double *out,
                       int direction) {
  const struct mock_fft_plan p = *plan;
  begin_operation(MOCK_FFT, p.stream, out, direction);
  const int total = p.dims[0] * p.dims[1] * p.dims[2] * p.batch;
  const double angle = direction * 6.283185307179586476925286766559;
  touch(in, 2 * sizeof(double) * (size_t)total);
  touch(out, 2 * sizeof(double) * (size_t)total);
  double *result = calloc(2 * (size_t)total, sizeof(double));
  assert(result != NULL);
  if (p.rank == 1) {
    const int n = p.dims[0];
    for (int b = 0; b < p.batch; b++) {
      for (int k = 0; k < n; k++) {
        const int dst = 2 * (b * p.odist + k * p.ostride);
        for (int j = 0; j < n; j++) {
          const int src = 2 * (b * p.idist + j * p.istride);
          const double c = cos(angle * j * k / n), s = sin(angle * j * k / n);
          result[dst] += in[src] * c - in[src + 1] * s;
          result[dst + 1] += in[src] * s + in[src + 1] * c;
        }
      }
    }
  } else {
    const int nx = p.dims[2], ny = p.dims[1], nz = p.dims[0];
    for (int k = 0; k < total; k++) {
      for (int j = 0; j < total; j++) {
        const double phase = angle * ((double)(j % nx) * (k % nx) / nx +
            (double)(j / nx % ny) * (k / nx % ny) / ny +
            (double)(j / (nx * ny)) * (k / (nx * ny)) / nz);
        const double c = cos(phase), s = sin(phase);
        result[2 * k] += in[2 * j] * c - in[2 * j + 1] * s;
        result[2 * k + 1] += in[2 * j] * s + in[2 * j + 1] * c;
      }
    }
  }
  memcpy(out, result, 2 * sizeof(double) * (size_t)total);
  free(result);
  mock_fft_calls++;
}

void pw_gpu_launch_real_to_complex(const double *in, double *out, int n,
                                   offloadStream_t stream) {
  assert(stream == 1);
  begin_operation(MOCK_REAL_TO_COMPLEX, stream, out, 0);
  touch(in, sizeof(double) * (size_t)n);
  touch(out, 2 * sizeof(double) * (size_t)n);
  for (int i = 0; i < n; i++) { out[2 * i] = in[i]; out[2 * i + 1] = 0; }
}
void pw_gpu_launch_complex_to_real(const double *in, double *out, int n,
                                   offloadStream_t stream) {
  assert(stream == 1);
  begin_operation(MOCK_COMPLEX_TO_REAL, stream, out, 0);
  touch(in, 2 * sizeof(double) * (size_t)n);
  touch(out, sizeof(double) * (size_t)n);
  for (int i = 0; i < n; i++) out[i] = in[2 * i];
}
void pw_gpu_launch_gather(double *out, const double *in, double scale,
                          int ngpts, const int *map, offloadStream_t stream) {
  assert(stream == 1);
  begin_operation(MOCK_GATHER, stream, out, 0);
  touch(out, 2 * sizeof(double) * (size_t)ngpts);
  touch(map, sizeof(int) * (size_t)ngpts);
  for (int i = 0; i < ngpts; i++) {
    touch(in, 2 * sizeof(double) * (size_t)(map[i] + 1));
    out[2 * i] = scale * in[2 * map[i]];
    out[2 * i + 1] = scale * in[2 * map[i] + 1];
  }
}
void pw_gpu_launch_scatter(double *out, const double *in, double scale,
                           int ngpts, int nmaps, const int *map,
                           offloadStream_t stream) {
  assert(stream == 1);
  begin_operation(MOCK_SCATTER, stream, out, 0);
  touch(in, 2 * sizeof(double) * (size_t)ngpts);
  touch(map, sizeof(int) * (size_t)ngpts * nmaps);
  for (int m = 0; m < nmaps; m++) {
    for (int i = 0; i < ngpts; i++) {
      int dst = map[m * ngpts + i];
      touch(out, 2 * sizeof(double) * (size_t)(dst + 1));
      out[2 * dst] = scale * in[2 * i];
      out[2 * dst + 1] = (m == 0 ? scale : -scale) * in[2 * i + 1];
    }
  }
}
void pw_gpu_launch_scatter_deriv(double *out, const double *in, const double *g,
                                 int idir, double scale, int ngpts, int nmaps,
                                 const int *map, offloadStream_t stream) {
  begin_operation(MOCK_SCATTER_DERIV, stream, out, idir);
  touch(in, 2 * sizeof(double) * (size_t)ngpts);
  touch(g, 3 * sizeof(double) * (size_t)ngpts);
  double *derivative = malloc(2 * sizeof(double) * (size_t)ngpts);
  assert(derivative != NULL && stream == 1);
  for (int i = 0; i < ngpts; i++) {
    derivative[2 * i] = -(in[2 * i + 1] * g[3 * i + idir]);
    derivative[2 * i + 1] = in[2 * i] * g[3 * i + idir];
  }
  touch(map, sizeof(int) * (size_t)ngpts * nmaps);
  for (int m = 0; m < nmaps; m++) {
    for (int i = 0; i < ngpts; i++) {
      int dst = map[m * ngpts + i];
      touch(out, 2 * sizeof(double) * (size_t)(dst + 1));
      out[2 * dst] = scale * derivative[2 * i];
      out[2 * dst + 1] = (m == 0 ? scale : -scale) * derivative[2 * i + 1];
    }
  }
  free(derivative);
}
void pw_gpu_launch_gather_deriv_acc(double *acc, const double *in, const double *g,
                                    int idir, double scale, int ngpts,
                                    const int *map, offloadStream_t stream) {
  assert(stream == 1);
  begin_operation(MOCK_GATHER_DERIV, stream, in, idir);
  touch(acc, 2 * sizeof(double) * (size_t)ngpts);
  touch(g, 3 * sizeof(double) * (size_t)ngpts);
  touch(map, sizeof(int) * (size_t)ngpts);
  for (int i = 0; i < ngpts; i++) {
    touch(in, 2 * sizeof(double) * (size_t)(map[i] + 1));
    acc[2 * i] -= (scale * in[2 * map[i] + 1]) * g[3 * i + idir];
    acc[2 * i + 1] += (scale * in[2 * map[i]]) * g[3 * i + idir];
  }
}

void pw_gpu_launch_copy(const double *x, double *y, size_t nx, size_t ny,
    int cx, int cy, double alpha, offloadStream_t stream) {
  begin_operation(MOCK_GRID_COPY, stream, y, 0);
  if (nx) touch(x, nx * (size_t)cx * sizeof(double));
  touch(y, ny * (size_t)cy * sizeof(double));
  for (size_t i = 0; i < ny; i++)
    for (int c = 0; c < cy; c++)
      y[(size_t)cy * i + c] = i < nx
          ? pw_grid_scale(pw_grid_component(x, i, cx, c), alpha) : 0.0;
  mock_grid_calls++;
}

void pw_gpu_launch_axpy(const double *x, double *y, size_t nx, size_t ny,
    int cx, int cy, double alpha, double beta, offloadStream_t stream) {
  begin_operation(MOCK_GRID_AXPY, stream, y, 0);
  if (nx && alpha != 0.0) touch(x, nx * (size_t)cx * sizeof(double));
  touch(y, ny * (size_t)cy * sizeof(double));
  for (size_t i = 0; i < ny; i++)
    for (int c = 0; c < cy; c++) {
      const size_t j = (size_t)cy * i + c;
      y[j] = pw_grid_axpy(i < nx && alpha != 0.0
                            ? pw_grid_component(x, i, cx, c) : 0.0,
                          beta != 0.0 ? y[j] : 0.0,
                          i < nx ? alpha : 0.0, beta);
    }
  mock_grid_calls++;
}

void pw_gpu_launch_remap_copy(const double *x, double *y, const int *map,
    size_t n, int cx, int cy, int scatter, double alpha, offloadStream_t stream) {
  begin_operation(MOCK_REMAP_COPY, stream, y, 0);
  if (n) touch(map, n * sizeof(int));
  for (size_t i = 0; i < n; i++) {
    const size_t src = scatter ? i : (size_t)map[i] - 1;
    const size_t dst = scatter ? (size_t)map[i] - 1 : i;
    touch(x, (src + 1) * (size_t)cx * sizeof(double));
    touch(y, (dst + 1) * (size_t)cy * sizeof(double));
    for (int c = 0; c < cy; c++)
      y[(size_t)cy * dst + c] =
          pw_grid_scale(pw_grid_component(x, src, cx, c), alpha);
  }
  mock_grid_calls++;
}

void pw_gpu_launch_remap_axpy(const double *x, double *y, const int *map,
    size_t n, int cx, int cy, int scatter, double alpha, double beta,
    offloadStream_t stream) {
  begin_operation(MOCK_REMAP_AXPY, stream, y, 0);
  if (n) touch(map, n * sizeof(int));
  for (size_t i = 0; i < n; i++) {
    const size_t src = scatter ? i : (size_t)map[i] - 1;
    const size_t dst = scatter ? (size_t)map[i] - 1 : i;
    if (alpha != 0.0) touch(x, (src + 1) * (size_t)cx * sizeof(double));
    touch(y, (dst + 1) * (size_t)cy * sizeof(double));
    for (int c = 0; c < cy; c++) {
      const size_t j = (size_t)cy * dst + c;
      y[j] = pw_grid_axpy(alpha != 0.0 ? pw_grid_component(x, src, cx, c) : 0.0,
                          beta != 0.0 ? y[j] : 0.0, alpha, beta);
    }
  }
  mock_grid_calls++;
}

void pw_gpu_launch_modify_hartree(double *v, const double *g, const double *gsq,
    const double *r, const double *radii, const double *cv, size_t n,
    size_t na, size_t nr, double cut, double vol, offloadStream_t stream) {
  begin_operation(MOCK_HARTREE, stream, v, 0);
  touch(v, 2 * n * sizeof(double));
  touch(g, 3 * n * sizeof(double));
  touch(gsq, n * sizeof(double));
  if (na) touch(r, 3 * na * sizeof(double));
  if (nr) touch(radii, nr * sizeof(double));
  if (na && nr) touch(cv, na * nr * sizeof(double));
  for (size_t i = 0; i < n; i++)
    pw_hartree_elem(v, g, gsq, r, radii, cv, i, na, nr, cut, vol);
  mock_hartree_calls++;
}

void pw_gpu_launch_pbe_lda(const double *rho, const double *norm, double *e0,
    double *erho, double *enorm, size_t n, int grad, double epsilon,
    double kappa, double beta, double mu, double sc, double sx, offloadStream_t stream) {
  begin_operation(MOCK_PBE, stream, e0, grad);
  touch(rho, n * sizeof(double));
  touch(norm, n * sizeof(double));
  touch(e0, n * sizeof(double));
  if (grad) {
    touch(erho, n * sizeof(double));
    touch(enorm, n * sizeof(double));
  }
  for (size_t i = 0; i < n; i++)
    pw_pbe_lda_elem(rho, norm, e0, erho, enorm, i, grad, epsilon,
                    kappa, beta, mu, sc, sx);
  mock_pbe_calls++;
}

void pw_gpu_launch_norm3(const double *g0, const double *g1, const double *g2,
                         double *norm, size_t n, offloadStream_t s) {
  begin_operation(MOCK_NORM3, s, norm, 0);
  touch(g0, n * sizeof(double)); touch(g1, n * sizeof(double));
  touch(g2, n * sizeof(double)); touch(norm, n * sizeof(double));
  for (size_t i = 0; i < n; i++) norm[i] = pw_xc_norm3(g0[i], g1[i], g2[i]);
}

void pw_gpu_launch_combine_div_input(double *g0, double *g1, double *g2,
    const double *vnorm, const double *norm, size_t n, double cutoff, offloadStream_t s) {
  begin_operation(MOCK_COMBINE_DIV, s, g0, 0);
  touch(g0, n * sizeof(double)); touch(g1, n * sizeof(double));
  touch(g2, n * sizeof(double)); touch(norm, n * sizeof(double));
  touch(vnorm, n * sizeof(double));
  for (size_t i = 0; i < n; i++) {
    const double factor = pw_xc_div_factor(vnorm[i], norm[i], cutoff);
    g0[i] = -g0[i] * factor; g1[i] = -g1[i] * factor; g2[i] = -g2[i] * factor;
  }
}

void pw_gpu_launch_add_vrho(double *vxc, const double *vrho, size_t n, offloadStream_t s) {
  begin_operation(MOCK_ADD_VRHO, s, vxc, 0);
  touch(vxc, n * sizeof(double)); touch(vrho, n * sizeof(double));
  for (size_t i = 0; i < n; i++) vxc[i] = vrho[i] + vxc[i];
}

void pw_gpu_launch_xc_cutoff(double *values, const double *rho, size_t n,
    double cutoff, double range, offloadStream_t s) {
  if (!n || range <= 0.0) return;
  begin_operation(MOCK_XC_CUTOFF, s, values, 0);
  touch(values, n * sizeof(double)); touch(rho, n * sizeof(double));
  for (size_t i = 0; i < n; i++) values[i] = pw_xc_cutoff(values[i], rho[i], cutoff, range);
}

void pw_gpu_launch_xc_reduce(const double *values, double *partial, size_t n, offloadStream_t s) {
  const size_t columns = n / PW_GPU_XC_REDUCE_CHUNK + (n % PW_GPU_XC_REDUCE_CHUNK != 0);
  begin_operation(MOCK_XC_REDUCE, s, partial, 0);
  touch(values, n * sizeof(double)); touch(partial, pw_gpu_xc_reduce_size(n) * sizeof(double));
  for (size_t i = 0; i < columns; i++) pw_xc_sum_column(values, partial, n, i, columns);
}

#include "pw_gpu_res_ops.h"
int mock_res_calls;
void pw_gpu_launch_res_cmul(double *out, const double *a, const double *w,
                            int width, size_t n, offloadStream_t s) {
  if (!n) return;
  begin_operation(MOCK_GRID_AXPY, s, out, 0);
  touch(out, 2 * n * sizeof(double)); touch(a, 2 * n * sizeof(double));
  touch(w, (size_t)width * n * sizeof(double));
  for (size_t i = 0; i < n; i++) pw_res_cmul_elem(out, a, w, width, i);
  mock_res_calls++;
}
void pw_gpu_launch_res_cdot_terms(const double *a, const double *b, double *p,
                                  size_t n, offloadStream_t s) {
  if (!n) return;
  begin_operation(MOCK_GRID_COPY, s, p, 0);
  touch(a, 2 * n * sizeof(double)); touch(b, 2 * n * sizeof(double));
  touch(p, n * sizeof(double));
  for (size_t i = 0; i < n; i++) p[i] = pw_res_cdot_term(a, b, i);
  mock_res_calls++;
}
void pw_gpu_launch_res_sample(const double *x, double *out, size_t n, int width,
                              offloadStream_t s) {
  const size_t count = pw_res_sample_count(n);
  if (!count) return;
  begin_operation(MOCK_GRID_COPY, s, out, 0);
  touch(x, (size_t)width * n * sizeof(double));
  touch(out, (size_t)width * count * sizeof(double));
  for (size_t k = 0; k < count; k++) {
    const size_t e = pw_res_sample_index(k, n);
    for (int c = 0; c < width; c++) out[(size_t)width * k + c] = x[(size_t)width * e + c];
  }
  mock_res_calls++;
}
// Stage 4: natural FFT order of XC.
static size_t mock_gt_doubles(int mode, size_t n0, size_t n1, size_t n) {
  return mode == 2 ? 3 * n : n0 + n1 + n / (n0 * n1);
}
size_t mock_nat_deriv_n, mock_nat_deriv_acc_n;
void pw_gpu_launch_nat_deriv(double *c, const double *z, const double *gt,
    int mode, size_t n0, size_t n1, int idir, double scale, size_t n,
    offloadStream_t s) {
  if (!n) return;
  assert(mode == 1 || mode == 2);
  begin_operation(MOCK_GRID_COPY, s, c, idir);
  touch(c, 2 * n * sizeof(double)); touch(z, 2 * n * sizeof(double));
  touch(gt, mock_gt_doubles(mode, n0, n1, n) * sizeof(double));
  for (size_t p = 0; p < n; p++)
    pw_nat_deriv_elem(c, z, pw_nat_gd(gt, mode, n0, n1, p, idir), scale, p);
  mock_nat_deriv_n += n;
  mock_res_calls++;
}
void pw_gpu_launch_nat_deriv_acc(double *acc, const double *c, const double *gt,
    int mode, size_t n0, size_t n1, int idir, double scale, size_t n,
    offloadStream_t s) {
  if (!n) return;
  assert(mode == 1 || mode == 2);
  begin_operation(MOCK_GRID_AXPY, s, acc, idir);
  touch(acc, 2 * n * sizeof(double)); touch(c, 2 * n * sizeof(double));
  touch(gt, mock_gt_doubles(mode, n0, n1, n) * sizeof(double));
  for (size_t p = 0; p < n; p++)
    pw_nat_deriv_acc_elem(acc, c, pw_nat_gd(gt, mode, n0, n1, p, idir), scale, p);
  mock_nat_deriv_acc_n += n;
  mock_res_calls++;
}
void pw_gpu_launch_gtab_build(double *gt, const double *g, const int *map,
    size_t n0, size_t n1, size_t n, offloadStream_t s) {
  if (!n) return;
  begin_operation(MOCK_GRID_COPY, s, gt, 0);
  touch(g, 3 * n * sizeof(double)); touch(map, n * sizeof(int));
  touch(gt, mock_gt_doubles(1, n0, n1, n) * sizeof(double));
  for (size_t i = 0; i < n; i++) {
    const size_t p = (size_t)map[i], r = p / n0;
    assert(p < n);
    gt[p % n0] = g[3 * i];
    gt[n0 + r % n1] = g[3 * i + 1];
    gt[n0 + n1 + r / n1] = g[3 * i + 2];
  }
  mock_res_calls++;
}
void pw_gpu_launch_gtab_check(double *flags, const double *g, const int *map,
    const double *gt, size_t n0, size_t n1, size_t n, offloadStream_t s) {
  if (!n) return;
  begin_operation(MOCK_GRID_COPY, s, flags, 0);
  touch(flags, n * sizeof(double)); touch(g, 3 * n * sizeof(double));
  touch(map, n * sizeof(int)); touch(gt, mock_gt_doubles(1, n0, n1, n) * sizeof(double));
  for (size_t i = 0; i < n; i++) {
    const size_t p = (size_t)map[i];
    int bad = 0;
    for (int d = 0; d < 3; d++)
      bad |= pw_res_bits_differ(g[3 * i + d], pw_nat_gd(gt, 1, n0, n1, p, d));
    flags[i] = bad ? 1.0 : 0.0;
  }
  mock_res_calls++;
}
// Stage 5: paired natural XC kernels and the fused Hartree begin.
size_t mock_nat_grad2_n, mock_nat_div2_n, mock_tot_cmul_n;
void pw_gpu_launch_nat_grad2(double *z, const double *rho, const double *gt, int mode,
    size_t n0, size_t n1, size_t n2, int a, int b, size_t n, offloadStream_t s) {
  if (!n) return;
  assert(mode == 1 || mode == 2);
  assert(n == n0 * n1 * n2 && z != rho);
  begin_operation(MOCK_GRID_COPY, s, z, a);
  touch(z, 2 * n * sizeof(double)); touch(rho, 2 * n * sizeof(double));
  touch(gt, mock_gt_doubles(mode, n0, n1, n) * sizeof(double));
  for (size_t p = 0; p < n; p++) pw_nat_grad2_elem(z, rho, gt, mode, n0, n1, n2, a, b, p);
  mock_nat_grad2_n += n;
  mock_res_calls++;
}
size_t mock_rho_dz_n;
void pw_gpu_launch_nat_rho_dz(double *z, const double *rho, const double *gt, int mode,
    size_t n0, size_t n1, size_t n2, size_t n, offloadStream_t s) {
  if (!n) return;
  assert(mode == 1 || mode == 2);
  assert(n == n0 * n1 * n2 && z != rho);
  begin_operation(MOCK_GRID_COPY, s, z, 0);
  touch(z, 2 * n * sizeof(double)); touch(rho, 2 * n * sizeof(double));
  touch(gt, mock_gt_doubles(mode, n0, n1, n) * sizeof(double));
  for (size_t p = 0; p < n; p++) pw_nat_rho_dz_elem(z, rho, gt, mode, n0, n1, n2, p);
  mock_rho_dz_n += n;
  mock_res_calls++;
}
void pw_gpu_launch_nat_div2(double *acc, const double *w, const double *gt, int mode,
    size_t n0, size_t n1, size_t n2, int a, int b, double scale, int accumulate,
    size_t n, offloadStream_t s) {
  if (!n) return;
  assert(mode == 1 || mode == 2);
  assert(n == n0 * n1 * n2 && acc != w);
  begin_operation(MOCK_GRID_AXPY, s, acc, a);
  touch(acc, 2 * n * sizeof(double)); touch(w, 2 * n * sizeof(double));
  touch(gt, mock_gt_doubles(mode, n0, n1, n) * sizeof(double));
  for (size_t p = 0; p < n; p++)
    pw_nat_div2_elem(acc, w, gt, mode, n0, n1, n2, a, b, scale, accumulate, p);
  mock_nat_div2_n += n;
  mock_res_calls++;
}
void pw_gpu_launch_pack2(const double *x, const double *y, double *c, size_t n,
    offloadStream_t s) {
  if (!n) return;
  begin_operation(MOCK_REAL_TO_COMPLEX, s, c, 0);
  touch(x, n * sizeof(double)); touch(y, n * sizeof(double)); touch(c, 2 * n * sizeof(double));
  for (size_t p = 0; p < n; p++) { c[2 * p] = x[p]; c[2 * p + 1] = y[p]; }
}
void pw_gpu_launch_unpack2(const double *c, double *x, double *y, size_t n,
    offloadStream_t s) {
  if (!n) return;
  begin_operation(MOCK_COMPLEX_TO_REAL, s, x, 0);
  touch(x, n * sizeof(double)); touch(y, n * sizeof(double)); touch(c, 2 * n * sizeof(double));
  for (size_t p = 0; p < n; p++) { x[p] = c[2 * p]; y[p] = c[2 * p + 1]; }
}
void pw_gpu_launch_res_tot_cmul(double *rho_tot, double *out, const double *core,
    const double *rg, const double *w, int width, size_t n, offloadStream_t s) {
  if (!n) return;
  begin_operation(MOCK_GRID_COPY, s, out, 0);
  touch(rho_tot, 2 * n * sizeof(double)); touch(out, 2 * n * sizeof(double));
  touch(core, 2 * n * sizeof(double)); touch(rg, 2 * n * sizeof(double));
  touch(w, (size_t)width * n * sizeof(double));
  for (size_t i = 0; i < n; i++) pw_res_tot_cmul_elem(rho_tot, out, core, rg, w, width, i);
  mock_tot_cmul_n += n;
  mock_res_calls++;
}
void pw_gpu_launch_gtab_full(double *gt, const double *g, const int *map,
    size_t n, offloadStream_t s) {
  if (!n) return;
  begin_operation(MOCK_GRID_COPY, s, gt, 0);
  touch(g, 3 * n * sizeof(double)); touch(map, n * sizeof(int));
  touch(gt, 3 * n * sizeof(double));
  for (size_t i = 0; i < n; i++) {
    const size_t p = (size_t)map[i];
    assert(p < n);
    for (int d = 0; d < 3; d++) gt[3 * p + (size_t)d] = g[3 * i + (size_t)d];
  }
  mock_res_calls++;
}
