/* SPDX-License-Identifier: GPL-2.0-or-later */
#include "mock_offload.h"
#include "pw_gpu_kernels.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

size_t mock_allocated, mock_peak, mock_limit, mock_allocations;
int mock_fft_calls, mock_uploads, mock_live_plans;
int mock_live_streams, mock_live_events, mock_event_records, mock_event_waits;
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
  mock_event_records = mock_event_waits = 0;
}

static void touch(const void *ptr, size_t bytes) {
  for (size_t i = 0; i < 64; i++) {
    if (allocations[i].ptr == ptr) {
      assert(ptr != NULL && bytes <= allocations[i].size);
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
}
void offloadMemcpyAsyncDtoH(void *dst, const void *src, size_t bytes,
                           offloadStream_t stream) {
  begin_operation(MOCK_DOWNLOAD, stream, src, 0);
  touch(src, bytes);
  memcpy(dst, src, bytes);
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
