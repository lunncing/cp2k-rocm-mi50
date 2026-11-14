PROGRAM test_cholesky
   USE cp_fm_cholesky
   USE cp_fm_types
   USE cholesky_test_support
   USE ISO_C_BINDING, ONLY: C_INT
!$ USE omp_lib, ONLY: omp_get_max_threads, omp_set_num_threads
   IMPLICIT NONE
   INTERFACE
      FUNCTION gate() BIND(C, name="pw_gpu_cholesky_fast_single_rank_c") RESULT(enabled)
         IMPORT :: C_INT
         INTEGER(C_INT) :: enabled
      END FUNCTION
      FUNCTION get_threads() BIND(C, name="openblas_get_num_threads") RESULT(threads)
         IMPORT :: C_INT
         INTEGER(C_INT) :: threads
      END FUNCTION
      SUBROUTINE set_threads(threads) BIND(C, name="openblas_set_num_threads")
         IMPORT :: C_INT
         INTEGER(C_INT), VALUE :: threads
      END SUBROUTINE
      FUNCTION thread_sets() BIND(C, name="mock_cholesky_thread_sets_c") RESULT(sets)
         IMPORT :: C_INT
         INTEGER(C_INT) :: sets
      END FUNCTION
      FUNCTION max_threads() BIND(C, name="mock_cholesky_max_threads_c") RESULT(threads)
         IMPORT :: C_INT
         INTEGER(C_INT) :: threads
      END FUNCTION
      FUNCTION min_threads() BIND(C, name="mock_cholesky_min_threads_c") RESULT(threads)
         IMPORT :: C_INT
         INTEGER(C_INT) :: threads
      END FUNCTION
      SUBROUTINE reset_threads() BIND(C, name="mock_cholesky_reset_threads_c")
      END SUBROUTINE
#if defined(__TEST_HIP)
      FUNCTION gpu_calls() BIND(C, name="mock_cholesky_gpu_calls_c") RESULT(calls)
         IMPORT :: C_INT
         INTEGER(C_INT) :: calls
      END FUNCTION
      SUBROUTINE fail_gpu_solve(successful_solves) BIND(C, name="mock_cholesky_fail_solve_c")
         IMPORT :: C_INT
         INTEGER(C_INT), VALUE :: successful_solves
      END SUBROUTINE
      FUNCTION gpu_failure_pending() BIND(C, name="mock_cholesky_failure_pending_c") RESULT(pending)
         IMPORT :: C_INT
         INTEGER(C_INT) :: pending
      END FUNCTION
      SUBROUTINE gpu_init() BIND(C, name="pw_gpu_init")
      END SUBROUTINE
      SUBROUTINE gpu_finalize() BIND(C, name="pw_gpu_finalize")
      END SUBROUTINE
      SUBROUTINE reset_gpu_trace() BIND(C, name="mock_reset_trace")
      END SUBROUTINE
#endif
   END INTERFACE
   REAL(dp), PARAMETER :: padding = -9123.25_dp
   INTEGER :: mode, order, itype, solve, right, transpose_case, expected_threads
   LOGICAL :: enabled, active_parallel = .FALSE.

   enabled = gate() /= 0
   expected_threads = 1
!$ CALL omp_set_num_threads(4)
!$ expected_threads = omp_get_max_threads()
   CALL set_threads(3_C_INT)
#if defined(__TEST_HIP)
   CALL gpu_init()
   IF (enabled) THEN
      CALL reduction(512, 1, failed_solve=0)
      CALL reduction(512, 1, failed_solve=1)
   END IF
#endif
   DO mode = 0, 4
      CALL factorization(mode)
   END DO
   DO order = 8, 512, 504
      DO itype = 1, 3
         CALL reduction(order, itype)
      END DO
      DO solve = 0, 1
         DO right = 0, 1
            DO transpose_case = 0, 3
               CALL restoration(order, solve, right, transpose_case)
            END DO
         END DO
      END DO
   END DO
!$ active_parallel = .TRUE.
!$OMP PARALLEL NUM_THREADS(2)
!$OMP SINGLE
   CALL factorization(0)
   CALL reduction(8, 1)
   CALL restoration(8, 1, 0, 0)
!$OMP END SINGLE
!$OMP END PARALLEL
#if defined(__TEST_HIP)
   CALL verify((gpu_calls() > 0) .EQV. enabled, "Incorrect GPU dispatch")
   CALL gpu_finalize()
#endif
   CALL verify(timers == 0 .AND. get_threads() == 3, "Leaked timer or BLAS thread count")
   PRINT *, "PASS Fortran Cholesky numerics, dispatch, padding, OpenMP and DLA-Future"
CONTAINS
   SUBROUTINE create(matrix, rows, columns, extra, layout_mode)
      TYPE(cp_fm_type), INTENT(OUT) :: matrix
      INTEGER, INTENT(IN) :: rows, columns, extra
      INTEGER, INTENT(IN), OPTIONAL :: layout_mode
      INTEGER :: leading
      leading = rows
#if defined(__parallel)
      leading = rows + extra
#endif
      ALLOCATE (matrix%matrix_struct, matrix%local_data(leading, columns))
      matrix%matrix_struct%nrow_global = rows
      matrix%matrix_struct%ncol_global = columns
      matrix%matrix_struct%descriptor = 0
      matrix%matrix_struct%descriptor(9) = leading
      matrix%local_data = padding
      IF (PRESENT(layout_mode)) THEN
         IF (layout_mode == 1) matrix%matrix_struct%para_env%num_pe = 2
         IF (layout_mode == 2) matrix%matrix_struct%context%num_pe = [1, 2]
         IF (layout_mode == 3) matrix%matrix_struct%context%num_pe = [2, 1]
      END IF
   END SUBROUTINE

   SUBROUTINE release(matrix)
      TYPE(cp_fm_type), INTENT(INOUT) :: matrix
      INTEGER :: rows
      rows = matrix%matrix_struct%nrow_global
      CALL verify(ALL(matrix%local_data(rows + 1:, :) == padding), "Modified leading-dimension padding")
      DEALLOCATE (matrix%local_data, matrix%matrix_struct)
   END SUBROUTINE

   SUBROUTINE upper_factor(factor, order)
      INTEGER, INTENT(IN) :: order
      REAL(dp), INTENT(OUT) :: factor(order, order)
      INTEGER :: row
      factor = 0.0_dp
      DO row = 1, order
         factor(row, row) = 2.0_dp + 0.001_dp*row
         IF (row < order) factor(row, row + 1) = 0.13_dp
         IF (row + 1 < order) factor(row, row + 2) = -0.07_dp
      END DO
   END SUBROUTINE

   SUBROUTINE check_dispatch(previous, fast)
      INTEGER, INTENT(IN) :: previous
      LOGICAL, INTENT(IN) :: fast
#if defined(__parallel)
      CALL verify(distributed_calls == previous + MERGE(0, 1, fast), "Incorrect single-rank/PBLAS dispatch")
#else
      CALL verify(distributed_calls == previous, "Serial build called PBLAS")
#endif
      CALL verify(get_threads() == 3, "BLAS threads were not restored")
   END SUBROUTINE

   SUBROUTINE factorization(layout_mode)
      INTEGER, INTENT(IN) :: layout_mode
      TYPE(cp_fm_type) :: matrix, rhs
      REAL(dp) :: factor(8, 8), original(8, 8), expected(8, 3), inverse(8, 8)
      INTEGER :: info, previous, old_dlaf, row, column
      LOGICAL :: fast
      CALL create(matrix, 9, 9, 3, layout_mode)
      CALL create(rhs, 8, 3, 5, layout_mode)
      CALL upper_factor(factor, 8)
      original = MATMUL(TRANSPOSE(factor), factor)
      matrix%local_data(1:8, 1:8) = original
      fast = enabled .AND. layout_mode == 0 .AND. .NOT. active_parallel
      cholesky_type = FM_CHOLESKY_TYPE_SCALAPACK
      IF (layout_mode == 4) cholesky_type = FM_CHOLESKY_TYPE_DLAF
      previous = distributed_calls
      old_dlaf = dlaf_calls
      CALL reset_threads()
      CALL cp_fm_cholesky_decompose(matrix, n=8, info_out=info)
      CALL verify(info == 0, "DPOTRF failed")
      IF (layout_mode == 4) THEN
#if defined(__DLAF) && defined(__parallel)
         CALL verify(dlaf_calls == old_dlaf + 1, "DLA-Future decomposition bypassed")
#elif defined(__parallel)
         CALL verify(distributed_calls == previous + 1, "DLA-Future selection bypassed ScaLAPACK")
#endif
      ELSE
         CALL check_dispatch(previous, fast)
      END IF
#if defined(__parallel)
      IF (fast) THEN
         CALL verify(max_threads() == MAX(3, expected_threads), "Incorrect temporary OpenBLAS thread count")
         CALL verify(min_threads() == MIN(3, expected_threads), "Incorrect temporary OpenBLAS minimum thread count")
      END IF
#endif
      DO column = 1, 8
         CALL verify(MAXVAL(ABS(matrix%local_data(1:column, column) - factor(1:column, column))) < 1e-12_dp, "Bad factor")
      END DO
      CALL verify(ALL(matrix%local_data(9, :) == padding), "Optional order overwrote extra row")
      DO column = 1, 3
         DO row = 1, 8
            expected(row, column) = SIN(0.13_dp*row + 0.21_dp*column)
         END DO
      END DO
      rhs%local_data(1:8, :) = MATMUL(original, expected)
      previous = distributed_calls
      CALL cp_fm_cholesky_solve(matrix, rhs, n=8, info_out=info)
      CALL check_dispatch(previous, enabled .AND. (layout_mode == 0 .OR. layout_mode == 4) .AND. .NOT. active_parallel)
      CALL verify(info == 0 .AND. MAXVAL(ABS(rhs%local_data(1:8, :) - expected)) < 1e-12_dp, "Bad solve")
      inverse = factor
      CALL dpotri('U', 8, inverse, 8, info)
      previous = distributed_calls
      old_dlaf = dlaf_calls
      CALL cp_fm_cholesky_invert(matrix, n=8, info_out=info)
      IF (layout_mode == 4) THEN
#if defined(__DLAF) && defined(__parallel)
         CALL verify(dlaf_calls == old_dlaf + 1, "DLA-Future inversion bypassed")
#elif defined(__parallel)
         CALL verify(distributed_calls == previous + 1, "DLA-Future inversion fallback bypassed")
#endif
      ELSE
         CALL check_dispatch(previous, fast)
      END IF
      DO column = 1, 8
         CALL verify(MAXVAL(ABS(matrix%local_data(1:column, column) - inverse(1:column, column))) < 1e-12_dp, "Bad inverse")
      END DO
      matrix%local_data(1, 1) = -1.0_dp
      CALL cp_fm_cholesky_decompose(matrix, n=8, info_out=info)
      CALL verify(info == 1 .AND. get_threads() == 3, "DPOTRF error handling or thread restoration")
      cholesky_type = FM_CHOLESKY_TYPE_SCALAPACK
      CALL release(matrix)
      CALL release(rhs)
   END SUBROUTINE

   SUBROUTINE reduction(order, itype, failed_solve)
      INTEGER, INTENT(IN) :: order, itype
      INTEGER, INTENT(IN), OPTIONAL :: failed_solve
      TYPE(cp_fm_type) :: matrix, triangular
      REAL(dp) :: expected(order, order), factor(order, order)
      INTEGER :: row, column, info, previous
#if defined(__TEST_HIP)
      INTEGER :: previous_gpu_calls
#endif
      CALL create(matrix, order, order, 3)
      CALL create(triangular, order, order, 5)
      CALL upper_factor(factor, order)
      triangular%local_data(1:order, 1:order) = factor
      DO column = 1, order
         DO row = 1, column
            expected(row, column) = COS(0.01_dp*(row + column))
            matrix%local_data(row, column) = expected(row, column)
         END DO
      END DO
      CALL dsygst(itype, 'U', order, expected, order, factor, order, info)
      CALL verify(info == 0, "DSYGST reference failed")
      previous = distributed_calls
      CALL reset_threads()
#if defined(__TEST_HIP)
      previous_gpu_calls = gpu_calls()
      IF (PRESENT(failed_solve)) CALL fail_gpu_solve(INT(failed_solve, C_INT))
#endif
      CALL cp_fm_cholesky_reduce(matrix, triangular, itype)
#if defined(__TEST_HIP)
      IF (PRESENT(failed_solve)) THEN
         CALL verify(gpu_failure_pending() == 0, "GPU reduce failure was not reached")
         CALL verify(gpu_calls() == previous_gpu_calls + failed_solve, "Wrong GPU reduce solve failed")
         CALL verify(thread_sets() == 2, "Failed GPU reduce did not enter the threaded CPU fallback")
         CALL verify(max_threads() == MAX(3, expected_threads), "Wrong reduce fallback maximum thread count")
         CALL verify(min_threads() == MIN(3, expected_threads), "Wrong reduce fallback minimum thread count")
      END IF
      CALL reset_gpu_trace()
#endif
      CALL check_dispatch(previous, enabled .AND. .NOT. active_parallel)
      DO column = 1, order
         CALL verify(MAXVAL(ABS(matrix%local_data(1:column, column) - expected(1:column, column))) < 2e-11_dp, "Bad reduce")
      END DO
#if defined(__parallel)
      IF (enabled .AND. .NOT. active_parallel .AND. (order < 512 .OR. itype /= 1)) THEN
         CALL verify(thread_sets() == 2 .AND. min_threads() == 1, "DSYGST did not use one thread and restore it")
      END IF
#endif
      CALL release(matrix)
      CALL release(triangular)
   END SUBROUTINE

   SUBROUTINE restoration(order, solve, right, transpose)
      INTEGER, INTENT(IN) :: order, solve, right, transpose
      TYPE(cp_fm_type) :: matrix, triangular, outm
      REAL(dp), ALLOCATABLE :: expected(:, :), factor(:, :)
      INTEGER :: columns, bdim, row, column, previous, old_copies
#if defined(__TEST_HIP)
      INTEGER :: previous_gpu_calls
#endif
      CHARACTER(8) :: operation
      CHARACTER :: position, trans
      CHARACTER, PARAMETER :: transposes(4) = ['N', 't', 'C', 'c']
      columns = MERGE(order/2, order, solve == 1)
      bdim = MERGE(columns, order, right == 1)
      CALL create(matrix, order, columns, 3)
      CALL create(triangular, bdim, bdim, 5)
      CALL create(outm, order, columns, 7)
      ALLOCATE (expected(order, columns), factor(bdim, bdim))
      CALL upper_factor(factor, bdim)
      triangular%local_data(1:bdim, 1:bdim) = factor
      DO column = 1, columns
         DO row = 1, order
            expected(row, column) = SIN(0.03_dp*row + 0.07_dp*column)
         END DO
      END DO
      matrix%local_data(1:order, :) = expected
      operation = MERGE('SOLVE   ', 'MULTIPLY', solve == 1)
      position = MERGE('R', 'L', right == 1)
      trans = transposes(transpose + 1)
      IF (solve == 1) THEN
         CALL dtrsm(position, 'U', trans, 'N', order, columns, 1.0_dp, factor, bdim, expected, order)
      ELSE
         CALL dtrmm(position, 'U', trans, 'N', order, columns, 1.0_dp, factor, bdim, expected, order)
      END IF
      previous = distributed_calls
      old_copies = copy_calls
      CALL reset_threads()
#if defined(__TEST_HIP)
      previous_gpu_calls = gpu_calls()
#endif
      CALL cp_fm_cholesky_restore(matrix, columns, triangular, outm, operation, position, trans)
#if defined(__TEST_HIP)
      IF (enabled .AND. .NOT. active_parallel .AND. order >= 512) THEN
         CALL verify(gpu_calls() == previous_gpu_calls + 1, "Restore did not reach the GPU BLAS dispatch")
         CALL verify(thread_sets() == 0, "GPU restore unexpectedly used the CPU fallback")
      ELSE
         CALL verify(gpu_calls() == previous_gpu_calls, "Unexpected GPU restore dispatch")
      END IF
      CALL reset_gpu_trace()
#endif
      CALL check_dispatch(previous, enabled .AND. .NOT. active_parallel)
      CALL verify(MAXVAL(ABS(outm%local_data(1:order, :) - expected)) < 2e-11_dp, "Bad restore transa="//trans)
#if defined(__parallel)
      IF (enabled .AND. .NOT. active_parallel) CALL verify(copy_calls == old_copies, "Fast restore used PDCOPY")
#endif
      CALL release(matrix)
      CALL release(triangular)
      CALL release(outm)
   END SUBROUTINE
END PROGRAM
