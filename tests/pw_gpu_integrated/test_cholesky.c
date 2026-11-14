#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "mock_cholesky.h"
#include "pw_gpu_internal.c"
#include <assert.h>
#include <math.h>
#include <stdio.h>

static double *matrix(int rows, int columns) {
  double *values = malloc((size_t)rows * columns * sizeof(double));
  assert(values != NULL);
  for (size_t index = 0; index < (size_t)rows * columns; index++) values[index] = -9123.25;
  return values;
}

static double factor(int row, int column) {
  if (row == column) return 2.0 + 0.001 * row;
  if (column == row + 1) return 0.13;
  if (column == row + 2) return -0.07;
  return 0.0;
}

static void seed_factor(double *values, int order, int leading) {
  for (int column = 0; column < order; column++) {
    for (int row = 0; row <= column; row++) values[row + (size_t)column * leading] = factor(row, column);
  }
}

static void product(int right, int transpose, int rows, int columns,
                    const double *input, int input_ld, double *out, int out_ld) {
  for (int column = 0; column < columns; column++) {
    for (int row = 0; row < rows; row++) {
      double value = 0.0;
      const int anchor = right ? column : row;
      for (int offset = -2; offset <= 2; offset++) {
        const int inner = anchor + offset;
        if (inner < 0 || inner >= (right ? columns : rows)) continue;
        const int brow = right ? inner : row, bcolumn = right ? column : inner;
        const double coefficient = transpose ? factor(bcolumn, brow) : factor(brow, bcolumn);
        value += coefficient * (right ? input[row + (size_t)inner * input_ld]
                                      : input[inner + (size_t)column * input_ld]);
      }
      out[row + (size_t)column * out_ld] = value;
    }
  }
}

static void close_matrix(const double *actual, int actual_ld, const double *expected,
                         int expected_ld, int rows, int columns) {
  for (int column = 0; column < columns; column++) {
    for (int row = 0; row < rows; row++) {
      const double want = expected[row + (size_t)column * expected_ld];
      assert(fabs(actual[row + (size_t)column * actual_ld] - want) < 2e-11 * (1.0 + fabs(want)));
    }
    for (int row = rows; row < actual_ld; row++) assert(actual[row + (size_t)column * actual_ld] == -9123.25);
  }
}

static void restore_cases(int enabled) {
  for (int padded = 0; padded <= 1; padded++) {
    for (int solve = 0; solve <= 1; solve++) {
      for (int right = 0; right <= 1; right++) {
        for (int transpose = 0; transpose <= 1; transpose++) {
          const int rows = 512, columns = solve ? 256 : 513, order = right ? columns : rows;
          const int lda = rows + 3 * padded, ldb = order + 5 * padded, ldout = rows + 7 * padded;
          double *input = matrix(lda, columns), *triangular = matrix(ldb, order);
          double *out = matrix(ldout, columns), *expected = matrix(rows, columns);
          double *seed = matrix(rows, columns);
          for (int column = 0; column < columns; column++) {
            for (int row = 0; row < rows; row++) seed[row + (size_t)column * rows] = sin(0.03 * row + 0.07 * column);
          }
          seed_factor(triangular, order, ldb);
          if (solve) {
            product(right, transpose, rows, columns, seed, rows, input, lda);
            memcpy(expected, seed, (size_t)rows * columns * sizeof(double));
          } else {
            for (int column = 0; column < columns; column++) {
              memcpy(input + (size_t)column * lda, seed + (size_t)column * rows, (size_t)rows * sizeof(double));
            }
            product(right, transpose, rows, columns, seed, rows, expected, rows);
          }
          double *original = matrix(lda, columns);
          memcpy(original, input, (size_t)lda * columns * sizeof(double));
          assert(pw_gpu_cholesky_restore_c(solve, right, transpose, rows, columns,
                                          triangular, ldb, input, lda, out, ldout) == enabled);
          assert(memcmp(original, input, (size_t)lda * columns * sizeof(double)) == 0);
          if (enabled) {
            close_matrix(out, ldout, expected, rows, rows, columns);
            assert(pw_gpu_cholesky_restore_c(solve, right, transpose, rows, columns,
                                            triangular, ldb, input, lda, input, lda) == 1);
            close_matrix(input, lda, expected, rows, rows, columns);
          } else {
            for (size_t index = 0; index < (size_t)ldout * columns; index++) assert(out[index] == -9123.25);
          }
          for (int column = 0; column < order; column++) {
            for (int row = 0; row < ldb; row++) {
              assert(triangular[row + (size_t)column * ldb] == (row <= column ? factor(row, column) : -9123.25));
            }
          }
          free(original);
          free(input);
          free(triangular);
          free(out);
          free(expected);
          free(seed);
          mock_assert_idle();
          mock_reset_trace();
          assert(mock_allocated == 0);
        }
      }
    }
  }
}

static void reduce_case(int enabled, int padded) {
  const int order = 512, lda = order + 3 * padded, ldb = order + 5 * padded;
  double *expected = matrix(order, order), *scratch = matrix(order, order);
  double *input = matrix(lda, order), *triangular = matrix(ldb, order);
  for (int column = 0; column < order; column++) {
    for (int row = 0; row < order; row++) expected[row + (size_t)column * order] = cos(0.01 * (row + column));
  }
  seed_factor(triangular, order, ldb);
  product(0, 1, order, order, expected, order, scratch, order);
  product(1, 0, order, order, scratch, order, input, lda);
  for (int column = 0; column < order; column++) {
    for (int row = column + 1; row < order; row++) input[row + (size_t)column * lda] = -9123.25;
  }
  double *original = matrix(lda, order);
  memcpy(original, input, (size_t)lda * order * sizeof(double));
  assert(pw_gpu_cholesky_reduce_c(order, input, lda, triangular, ldb) == enabled);
  if (enabled) close_matrix(input, lda, expected, order, order, order);
  else assert(memcmp(original, input, (size_t)lda * order * sizeof(double)) == 0);
  free(original);
  free(input);
  free(triangular);
  free(expected);
  free(scratch);
  mock_assert_idle();
  mock_reset_trace();
}

static void fallback_cases(void) {
  const int order = 512, leading = 515;
  const size_t bytes = (size_t)leading * order * sizeof(double);
  double *input = matrix(leading, order), *triangular = matrix(leading, order);
  double *out = matrix(leading, order), *original = matrix(leading, order);
  seed_factor(triangular, order, leading);
  memcpy(original, input, bytes);
  assert(!pw_gpu_cholesky_restore_c(1, 0, 0, 511, order, triangular, leading, input, leading, out, leading));
  assert(!pw_gpu_cholesky_restore_c(1, 0, 0, order, 255, triangular, leading, input, leading, out, leading));
  assert(!pw_gpu_cholesky_restore_c(0, 0, 0, order, 511, triangular, leading, input, leading, out, leading));
  assert(!pw_gpu_cholesky_restore_c(1, 0, 0, order, 0, triangular, leading, input, leading, out, leading));
  assert(!pw_gpu_cholesky_restore_c(1, 0, 0, order, order, triangular, order - 1, input, leading, out, leading));
  assert(!pw_gpu_cholesky_restore_c(1, 0, 0, order, order, triangular, leading, input, order - 1, out, leading));
  assert(!pw_gpu_cholesky_restore_c(1, 0, 0, order, order, triangular, leading, input, leading, out, order - 1));
  assert(!pw_gpu_cholesky_reduce_c(511, input, leading, triangular, leading));
#pragma omp parallel num_threads(2)
  {
    assert(!pw_gpu_cholesky_restore_c(1, 0, 0, order, order, triangular, leading, input, leading, out, leading));
    assert(!pw_gpu_cholesky_reduce_c(order, input, leading, triangular, leading));
  }
  pw_gpu_set_memory_limit(1);
  assert(!pw_gpu_cholesky_reduce_c(order, input, leading, triangular, leading));
  assert(!pw_gpu_cholesky_restore_c(1, 0, 0, order, order, triangular, leading, input, leading, out, leading));
  pw_gpu_set_memory_limit(0);
  mock_device_total_bytes = (size_t)64 * 1024 * 1024;
  assert(!pw_gpu_cholesky_reduce_c(order, input, leading, triangular, leading));
  assert(!pw_gpu_cholesky_restore_c(0, 0, 0, order, order, triangular, leading, input, leading, out, leading));
  mock_device_total_bytes = 0;
#if defined(__OFFLOAD_HIP) && !defined(CHOLESKY_NO_ROCBLAS)
  for (int operation = CHOLESKY_CREATE; operation <= CHOLESKY_SYNCHRONIZE; operation++) {
    pw_gpu_finalize();
    pw_gpu_init();
    mock_cholesky_failure = operation;
    mock_cholesky_fail_after = operation == CHOLESKY_DOWNLOAD || operation == CHOLESKY_UPLOAD ? 5 : 0;
    assert(!pw_gpu_cholesky_restore_c(operation != CHOLESKY_MULTIPLY, 0, 0, order, order,
                                    triangular, leading, input, leading, out, leading));
    assert(mock_cholesky_failure == 0);
    assert(memcmp(original, input, bytes) == 0 && memcmp(original, out, bytes) == 0);
    assert(mock_allocated == 0 && current_gpu_allocated_bytes == 0);
    mock_assert_idle();
    mock_reset_trace();
  }
  for (int second = 0; second <= 1; second++) {
    mock_cholesky_failure = CHOLESKY_SOLVE;
    mock_cholesky_fail_after = second;
    assert(!pw_gpu_cholesky_reduce_c(order, input, leading, triangular, leading));
    assert(memcmp(original, input, bytes) == 0 && mock_allocated == 0);
    mock_reset_trace();
  }
  mock_cholesky_failure = CHOLESKY_DOWNLOAD;
  mock_cholesky_fail_after = 5;
  assert(!pw_gpu_cholesky_reduce_c(order, input, leading, triangular, leading));
  assert(memcmp(original, input, bytes) == 0 && mock_allocated == 0);
  mock_reset_trace();
#endif
  free(input);
  free(triangular);
  free(out);
  free(original);
}

#if defined(__OFFLOAD_HIP) && !defined(CHOLESKY_NO_ROCBLAS)
static void budget_cases(void) {
  const int order = 512, leading = 515;
  double *input = matrix(leading, order), *triangular = matrix(leading, order);
  double *out = matrix(leading, order);
  seed_factor(triangular, order, leading);
  const size_t reserve = res_reserve_bytes ? res_reserve_bytes : 256 * PW_GPU_MIB;
  for (int solve = 0; solve <= 1; solve++) {
    const size_t required = (size_t)(solve ? 2 : 3) * order * order * sizeof(double);
    const size_t allocations_before = mock_allocations;
    max_gpu_memory_bytes = required - 1;
    pw_gpu_set_host_fallback(0);
    assert(!pw_gpu_cholesky_restore_c(solve, 0, 0, order, order, triangular, leading, input, leading, out, leading));
    pw_gpu_set_host_fallback(1);
    max_gpu_memory_bytes = required;
    mock_device_total_bytes = required + reserve - 1;
    assert(!pw_gpu_cholesky_restore_c(solve, 0, 0, order, order, triangular, leading, input, leading, out, leading));
    assert(mock_allocations == allocations_before);
    mock_device_total_bytes++;
    assert(pw_gpu_cholesky_restore_c(solve, 0, 0, order, order, triangular, leading, input, leading, out, leading));
    assert(mock_allocations == allocations_before + 1 && mock_allocated == 0 && current_gpu_allocated_bytes == 0);
    mock_reset_trace();
  }
  const size_t required = (size_t)2 * order * order * sizeof(double);
  max_gpu_memory_bytes = required - 1;
  assert(!pw_gpu_cholesky_reduce_c(order, input, leading, triangular, leading));
  max_gpu_memory_bytes++;
  mock_device_total_bytes = required + reserve - 1;
  assert(!pw_gpu_cholesky_reduce_c(order, input, leading, triangular, leading));
  mock_device_total_bytes++;
  assert(pw_gpu_cholesky_reduce_c(order, input, leading, triangular, leading));
  assert(mock_allocated == 0 && current_gpu_allocated_bytes == 0);
  max_gpu_memory_bytes = 0;
  mock_device_total_bytes = 0;
  mock_reset_trace();
  free(input);
  free(triangular);
  free(out);
}
#endif

int main(int argc, char **argv) {
  assert(argc == 2);
  const int enabled = atoi(argv[1]);
  const int gate = pw_gpu_cholesky_fast_single_rank_c();
  assert(pw_gpu_openblas_threads_begin_c(0) == 0);
  assert(pw_gpu_openblas_threads_begin_c(-1) == 0 && mock_blas_sets == 0);
  mock_blas_threads = 3;
  const int previous = pw_gpu_openblas_threads_begin_c(7);
#if defined(CHOLESKY_NO_BLAS_GET) || defined(CHOLESKY_NO_BLAS_SET)
  assert(previous == 0 && mock_blas_threads == 3);
#else
  assert(previous == 3 && mock_blas_threads == 7);
#endif
  pw_gpu_openblas_threads_end_c(previous);
  pw_gpu_openblas_threads_end_c(0);
  assert(mock_blas_threads == 3);
  assert(!pw_gpu_cholesky_restore_c(1, 0, 0, 512, 512, NULL, 512, NULL, 512, NULL, 512));
  assert(!pw_gpu_cholesky_reduce_c(512, NULL, 512, NULL, 512));
  pw_gpu_init();
  restore_cases(enabled);
  reduce_case(enabled, 0);
  reduce_case(enabled, 1);
  if (enabled) assert(mock_cholesky_creates == 1 && mock_cholesky_destroys == 0);
#if defined(__OFFLOAD_HIP) && !defined(CHOLESKY_NO_ROCBLAS)
  if (enabled) budget_cases();
#endif
  if (gate) fallback_cases();
  pw_gpu_finalize();
  assert(mock_cholesky_creates == mock_cholesky_destroys);
  assert(mock_allocated == 0 && mock_live_streams == 0 && mock_live_events == 0);
  if (enabled) {
    pw_gpu_init();
    reduce_case(1, 1);
    pw_gpu_finalize();
    assert(mock_cholesky_creates == mock_cholesky_destroys);
  }
  puts("PASS Cholesky C numerics, strides, aliasing, fallbacks, threads and lifecycle");
  return 0;
}
