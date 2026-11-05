/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef MOCK_OFFLOAD_H
#define MOCK_OFFLOAD_H
#include <stddef.h>

typedef int offloadStream_t;
typedef int offloadEvent_t;
typedef struct mock_fft_plan *offload_fftHandle;
#define OFFLOAD_FFT_FORWARD (-1)
#define OFFLOAD_FFT_INVERSE 1
#define OFFLOAD_FFT_Z2Z 0

extern size_t mock_allocated, mock_peak, mock_limit, mock_allocations;
extern int mock_fft_calls, mock_uploads, mock_live_plans;
extern int mock_live_streams, mock_live_events, mock_event_records, mock_event_waits;

enum mock_operation_kind {
  MOCK_UPLOAD, MOCK_DOWNLOAD, MOCK_MEMSET, MOCK_FFT, MOCK_REAL_TO_COMPLEX,
  MOCK_COMPLEX_TO_REAL, MOCK_GATHER, MOCK_SCATTER, MOCK_SCATTER_DERIV,
  MOCK_GATHER_DERIV
};
typedef struct {
  enum mock_operation_kind kind;
  offloadStream_t stream;
  const void *buffer;
  int direction;
  size_t clock[3];
} mock_operation;
extern mock_operation mock_operations[8192];
extern size_t mock_operation_count;
void mock_assert_idle(void);
void mock_reset_trace(void);

void offload_activate_chosen_device(void);
void offloadMalloc(void **ptr, size_t bytes);
void offloadFree(void *ptr);
void offloadStreamCreate(offloadStream_t *stream);
void offloadStreamDestroy(offloadStream_t stream);
void offloadStreamSynchronize(offloadStream_t stream);
void offloadEventCreate(offloadEvent_t *event);
void offloadEventDestroy(offloadEvent_t event);
void offloadEventRecord(offloadEvent_t event, offloadStream_t stream);
void offloadStreamWaitEvent(offloadStream_t stream, offloadEvent_t event);
void offloadMemcpyAsyncHtoD(void *dst, const void *src, size_t bytes,
                           offloadStream_t stream);
void offloadMemcpyAsyncDtoH(void *dst, const void *src, size_t bytes,
                           offloadStream_t stream);
void offloadMemsetAsync(void *dst, int value, size_t bytes,
                        offloadStream_t stream);
void offload_fftPlanMany(offload_fftHandle *plan, int rank, int *dims,
                         int *inembed, int istride, int idist, int *onembed,
                         int ostride, int odist, int type, int batch);
void offload_fftPlan3d(offload_fftHandle *plan, int nx, int ny, int nz, int type);
void offload_fftSetStream(offload_fftHandle plan, offloadStream_t stream);
void offload_fftDestroy(offload_fftHandle plan);
void offload_fftExecZ2Z(offload_fftHandle plan, const double *in, double *out,
                       int direction);
#endif
