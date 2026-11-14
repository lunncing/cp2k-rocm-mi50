#include "mock_cholesky.h"
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

int mock_cholesky_failure, mock_cholesky_fail_after;
int mock_cholesky_creates, mock_cholesky_destroys, mock_cholesky_solves, mock_cholesky_multiplies;
int mock_blas_threads = 1, mock_blas_sets, mock_blas_max_threads;
static int mock_blas_min_threads = INT_MAX;

int mock_cholesky_thread_sets_c(void) { return mock_blas_sets; }
int mock_cholesky_max_threads_c(void) { return mock_blas_max_threads; }
int mock_cholesky_min_threads_c(void) { return mock_blas_min_threads; }
int mock_cholesky_gpu_calls_c(void) { return mock_cholesky_solves + mock_cholesky_multiplies; }
void mock_cholesky_fail_solve_c(int successful_solves) {
  mock_cholesky_failure = CHOLESKY_SOLVE;
  mock_cholesky_fail_after = successful_solves;
}
int mock_cholesky_failure_pending_c(void) { return mock_cholesky_failure != 0; }
void mock_cholesky_reset_threads_c(void) {
  mock_blas_sets = 0;
  mock_blas_max_threads = 0;
  mock_blas_min_threads = INT_MAX;
}

static int fail(int operation) {
  if (mock_cholesky_failure != operation) return 0;
  if (mock_cholesky_fail_after-- > 0) return 0;
  mock_cholesky_failure = 0;
  return 1;
}

#if !defined(CHOLESKY_NO_BLAS_GET)
int openblas_get_num_threads(void) { return mock_blas_threads; }
#endif
#if !defined(CHOLESKY_NO_BLAS_SET)
void openblas_set_num_threads(int threads) {
  assert(threads > 0);
  mock_blas_threads = threads;
  mock_blas_sets++;
  if (threads > mock_blas_max_threads) mock_blas_max_threads = threads;
  if (threads < mock_blas_min_threads) mock_blas_min_threads = threads;
}
#endif

int hipMalloc(void **pointer, size_t bytes) {
  if (fail(CHOLESKY_ALLOCATE)) return 1;
  offloadMalloc(pointer, bytes);
  return hipSuccess;
}

int hipFree(void *pointer) {
  offloadFree(pointer);
  return hipSuccess;
}

int hipMemGetInfo(size_t *free_bytes, size_t *total_bytes) {
  offloadMemGetInfo(free_bytes, total_bytes);
  return hipSuccess;
}

int hipMemcpy2DAsync(void *destination, size_t destination_pitch,
                     const void *source, size_t source_pitch, size_t width,
                     size_t height, int kind, offloadStream_t stream) {
  assert(destination_pitch >= width && source_pitch >= width);
  for (size_t column = 0; column < height; column++) {
    if (fail(kind == hipMemcpyHostToDevice ? CHOLESKY_UPLOAD : CHOLESKY_DOWNLOAD)) return 1;
    void *target = (char *)destination + column * destination_pitch;
    const void *input = (const char *)source + column * source_pitch;
    if (kind == hipMemcpyHostToDevice) {
      offloadMemcpyAsyncHtoD(target, input, width, stream);
    } else {
      assert(kind == hipMemcpyDeviceToHost);
      offloadMemcpyAsyncDtoH(target, input, width, stream);
    }
  }
  return hipSuccess;
}

int hipStreamSynchronize(offloadStream_t stream) {
  offloadStreamSynchronize(stream);
  return fail(CHOLESKY_SYNCHRONIZE) ? 1 : hipSuccess;
}

#if !defined(CHOLESKY_NO_ROCBLAS)
struct _rocblas_handle { offloadStream_t stream; };
typedef struct _rocblas_handle *mock_rocblas_handle;

int rocblas_create_handle(mock_rocblas_handle *handle) {
  *handle = malloc(sizeof(**handle));
  assert(*handle != NULL);
  (*handle)->stream = 0;
  mock_cholesky_creates++;
  return fail(CHOLESKY_CREATE);
}

int rocblas_destroy_handle(mock_rocblas_handle handle) {
  assert(handle != NULL);
  free(handle);
  mock_cholesky_destroys++;
  return 0;
}

int rocblas_set_stream(mock_rocblas_handle handle, offloadStream_t stream) {
  handle->stream = stream;
  return fail(CHOLESKY_STREAM);
}

extern void dtrsm_(const char *, const char *, const char *, const char *, const int *, const int *,
                   const double *, const double *, const int *, double *, const int *);
extern void dtrmm_(const char *, const char *, const char *, const char *, const int *, const int *,
                   const double *, const double *, const int *, double *, const int *);

int rocblas_dtrsm(mock_rocblas_handle handle, int side, int uplo, int trans, int diag,
                  int rows, int columns, const double *alpha, const double *factor,
                  int factor_ld, double *matrix, int matrix_ld) {
  assert(handle->stream != 0 && uplo == 121 && diag == 131);
  assert(side == 141 || side == 142);
  assert(trans == 111 || trans == 112);
  assert(factor_ld == (side == 142 ? columns : rows) && matrix_ld == rows && *alpha == 1.0);
  if (fail(CHOLESKY_SOLVE)) return 1;
  mock_cholesky_solves++;
  const char position = side == 142 ? 'R' : 'L', transpose = trans == 112 ? 'T' : 'N';
  dtrsm_(&position, "U", &transpose, "N", &rows, &columns, alpha, factor, &factor_ld, matrix, &matrix_ld);
  return 0;
}

int rocblas_dtrmm(mock_rocblas_handle handle, int side, int uplo, int trans, int diag,
                  int rows, int columns, const double *alpha, const double *factor,
                  int factor_ld, const double *matrix, int matrix_ld, double *out, int out_ld) {
  assert(handle->stream != 0 && uplo == 121 && diag == 131);
  assert(side == 141 || side == 142);
  assert(trans == 111 || trans == 112);
  assert(factor_ld == (side == 142 ? columns : rows) && matrix_ld == rows && out_ld == rows);
  assert(out != matrix && *alpha == 1.0);
  if (fail(CHOLESKY_MULTIPLY)) return 1;
  mock_cholesky_multiplies++;
  memcpy(out, matrix, (size_t)rows * columns * sizeof(double));
  const char position = side == 142 ? 'R' : 'L', transpose = trans == 112 ? 'T' : 'N';
  dtrmm_(&position, "U", &transpose, "N", &rows, &columns, alpha, factor, &factor_ld, out, &out_ld);
  return 0;
}
#endif
