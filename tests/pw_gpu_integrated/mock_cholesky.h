#ifndef MOCK_CHOLESKY_H
#define MOCK_CHOLESKY_H
#include "mock_offload.h"

enum { hipSuccess = 0, hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2 };
int hipMalloc(void **pointer, size_t bytes);
int hipFree(void *pointer);
int hipMemGetInfo(size_t *free_bytes, size_t *total_bytes);
int hipMemcpy2DAsync(void *destination, size_t destination_pitch,
                     const void *source, size_t source_pitch, size_t width,
                     size_t height, int kind, offloadStream_t stream);
int hipStreamSynchronize(offloadStream_t stream);

extern int mock_cholesky_failure, mock_cholesky_fail_after;
extern int mock_cholesky_creates, mock_cholesky_destroys, mock_cholesky_solves, mock_cholesky_multiplies;
extern int mock_blas_threads, mock_blas_sets, mock_blas_max_threads;
enum {
  CHOLESKY_CREATE = 1, CHOLESKY_STREAM, CHOLESKY_ALLOCATE, CHOLESKY_UPLOAD,
  CHOLESKY_SOLVE, CHOLESKY_MULTIPLY, CHOLESKY_DOWNLOAD, CHOLESKY_SYNCHRONIZE
};
#endif
