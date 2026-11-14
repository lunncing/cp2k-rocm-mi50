MODULE kinds
   USE ISO_C_BINDING, ONLY: C_DOUBLE
   IMPLICIT NONE
   INTEGER, PARAMETER :: dp = C_DOUBLE
END MODULE

MODULE cholesky_test_support
   IMPLICIT NONE
   INTEGER :: distributed_calls = 0, dlaf_calls = 0, copy_calls = 0, timers = 0
CONTAINS
   SUBROUTINE verify(condition, message)
      LOGICAL, INTENT(IN) :: condition
      CHARACTER(*), INTENT(IN) :: message
      IF (.NOT. condition) THEN
         PRINT *, message
         ERROR STOP 1
      END IF
   END SUBROUTINE
   SUBROUTINE abort_test(message)
      CHARACTER(*), INTENT(IN) :: message
      PRINT *, message
      ERROR STOP 2
   END SUBROUTINE
   SUBROUTINE cp_abort(location, message)
      CHARACTER(*), INTENT(IN) :: location, message
      PRINT *, location, message
      ERROR STOP 3
   END SUBROUTINE
   SUBROUTINE timeset(name, handle)
      CHARACTER(*), INTENT(IN) :: name
      INTEGER, INTENT(OUT) :: handle
      timers = timers + 1
      handle = timers
   END SUBROUTINE
   SUBROUTINE timestop(handle)
      INTEGER, INTENT(IN) :: handle
      timers = timers - 1
   END SUBROUTINE
END MODULE

MODULE cp_fm_types
   USE kinds
   IMPLICIT NONE
   TYPE environment_type
      INTEGER :: num_pe = 1
   END TYPE
   TYPE context_type
      INTEGER :: num_pe(2) = 1
   CONTAINS
      PROCEDURE :: get_handle
   END TYPE
   TYPE layout_type
      INTEGER :: nrow_global, ncol_global, descriptor(9)
      TYPE(environment_type) :: para_env
      TYPE(context_type) :: context
   END TYPE
   TYPE cp_fm_type
      TYPE(layout_type), POINTER :: matrix_struct
      REAL(dp), POINTER :: local_data(:, :)
   END TYPE
CONTAINS
   INTEGER FUNCTION get_handle(this)
      CLASS(context_type), INTENT(IN) :: this
      get_handle = 1
   END FUNCTION
END MODULE

MODULE cp_log_handling
   USE kinds
   IMPLICIT NONE
   INTERFACE cp_to_string
      MODULE PROCEDURE integer_string, real_string
   END INTERFACE
CONTAINS
   FUNCTION integer_string(value) RESULT(text)
      INTEGER, INTENT(IN) :: value
      CHARACTER(32) :: text
      WRITE (text, *) value
   END FUNCTION
   FUNCTION real_string(value) RESULT(text)
      REAL(dp), INTENT(IN) :: value
      CHARACTER(32) :: text
      WRITE (text, *) value
   END FUNCTION
END MODULE

MODULE cp_dlaf_utils_api
   IMPLICIT NONE
CONTAINS
   SUBROUTINE cp_dlaf_initialize()
   END SUBROUTINE
   SUBROUTINE cp_dlaf_create_grid(handle)
      INTEGER, INTENT(IN) :: handle
   END SUBROUTINE
END MODULE

MODULE cp_fm_dlaf_api
   USE kinds
   USE cholesky_test_support
   IMPLICIT NONE
CONTAINS
   SUBROUTINE cp_pdpotrf_dlaf(uplo, order, matrix, row, column, descriptor, info)
      CHARACTER, INTENT(IN) :: uplo
      INTEGER, INTENT(IN) :: order, row, column, descriptor(9)
      INTEGER, INTENT(OUT) :: info
      REAL(dp), INTENT(INOUT) :: matrix(:, :)
      dlaf_calls = dlaf_calls + 1
      CALL dpotrf(uplo, order, matrix, SIZE(matrix, 1), info)
   END SUBROUTINE
   SUBROUTINE cp_pdpotri_dlaf(uplo, order, matrix, row, column, descriptor, info)
      CHARACTER, INTENT(IN) :: uplo
      INTEGER, INTENT(IN) :: order, row, column, descriptor(9)
      INTEGER, INTENT(OUT) :: info
      REAL(dp), INTENT(INOUT) :: matrix(:, :)
      dlaf_calls = dlaf_calls + 1
      CALL dpotri(uplo, order, matrix, SIZE(matrix, 1), info)
   END SUBROUTINE
END MODULE

SUBROUTINE pdpotrf(uplo, order, matrix, row, column, descriptor, info)
   USE kinds
   USE cholesky_test_support
   IMPLICIT NONE
   CHARACTER, INTENT(IN) :: uplo
   INTEGER, INTENT(IN) :: order, row, column, descriptor(9)
   INTEGER, INTENT(OUT) :: info
   REAL(dp), INTENT(INOUT) :: matrix(descriptor(9), *)
   distributed_calls = distributed_calls + 1
   CALL dpotrf(uplo, order, matrix, descriptor(9), info)
END SUBROUTINE

SUBROUTINE pdpotri(uplo, order, matrix, row, column, descriptor, info)
   USE kinds
   USE cholesky_test_support
   IMPLICIT NONE
   CHARACTER, INTENT(IN) :: uplo
   INTEGER, INTENT(IN) :: order, row, column, descriptor(9)
   INTEGER, INTENT(OUT) :: info
   REAL(dp), INTENT(INOUT) :: matrix(descriptor(9), *)
   distributed_calls = distributed_calls + 1
   CALL dpotri(uplo, order, matrix, descriptor(9), info)
END SUBROUTINE

SUBROUTINE pdpotrs(uplo, order, nrhs, matrix, row, column, descriptor, rhs, rhs_row, rhs_column, rhs_descriptor, info)
   USE kinds
   USE cholesky_test_support
   IMPLICIT NONE
   CHARACTER, INTENT(IN) :: uplo
   INTEGER, INTENT(IN) :: order, nrhs, row, column, descriptor(9), rhs_row, rhs_column, rhs_descriptor(9)
   INTEGER, INTENT(OUT) :: info
   REAL(dp), INTENT(IN) :: matrix(descriptor(9), *)
   REAL(dp), INTENT(INOUT) :: rhs(rhs_descriptor(9), *)
   distributed_calls = distributed_calls + 1
   CALL dpotrs(uplo, order, nrhs, matrix, descriptor(9), rhs, rhs_descriptor(9), info)
END SUBROUTINE

SUBROUTINE pdsygst(itype, uplo, order, matrix, row, column, descriptor, factor, brow, bcolumn, bdesc, scale, info)
   USE kinds
   USE cholesky_test_support
   IMPLICIT NONE
   CHARACTER, INTENT(IN) :: uplo
   INTEGER, INTENT(IN) :: itype, order, row, column, descriptor(9), brow, bcolumn, bdesc(9)
   INTEGER, INTENT(OUT) :: info
   REAL(dp), INTENT(INOUT) :: matrix(descriptor(9), *)
   REAL(dp), INTENT(IN) :: factor(bdesc(9), *)
   REAL(dp), INTENT(OUT) :: scale
   distributed_calls = distributed_calls + 1
   CALL dsygst(itype, uplo, order, matrix, descriptor(9), factor, bdesc(9), info)
   scale = 1.0_dp
END SUBROUTINE

SUBROUTINE pdcopy(order, matrix, row, column, descriptor, increment, outm, outrow, outcolumn, outdesc, outinc)
   USE kinds
   USE cholesky_test_support
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: order, row, column, descriptor(9), increment, outrow, outcolumn, outdesc(9), outinc
   REAL(dp), INTENT(IN) :: matrix(descriptor(9), *)
   REAL(dp), INTENT(INOUT) :: outm(outdesc(9), *)
   copy_calls = copy_calls + 1
   outm(1:order, outcolumn) = matrix(1:order, column)
END SUBROUTINE

SUBROUTINE pdtrsm(side, uplo, trans, diag, rows, columns, alpha, factor, brow, bcolumn, bdesc, outm, row, column, descriptor)
   USE kinds
   USE cholesky_test_support
   IMPLICIT NONE
   CHARACTER, INTENT(IN) :: side, uplo, trans, diag
   INTEGER, INTENT(IN) :: rows, columns, brow, bcolumn, bdesc(9), row, column, descriptor(9)
   REAL(dp), INTENT(IN) :: alpha, factor(bdesc(9), *)
   REAL(dp), INTENT(INOUT) :: outm(descriptor(9), *)
   distributed_calls = distributed_calls + 1
   CALL dtrsm(side, uplo, trans, diag, rows, columns, alpha, factor, bdesc(9), outm, descriptor(9))
END SUBROUTINE

SUBROUTINE pdtrmm(side, uplo, trans, diag, rows, columns, alpha, factor, brow, bcolumn, bdesc, outm, row, column, descriptor)
   USE kinds
   USE cholesky_test_support
   IMPLICIT NONE
   CHARACTER, INTENT(IN) :: side, uplo, trans, diag
   INTEGER, INTENT(IN) :: rows, columns, brow, bcolumn, bdesc(9), row, column, descriptor(9)
   REAL(dp), INTENT(IN) :: alpha, factor(bdesc(9), *)
   REAL(dp), INTENT(INOUT) :: outm(descriptor(9), *)
   distributed_calls = distributed_calls + 1
   CALL dtrmm(side, uplo, trans, diag, rows, columns, alpha, factor, bdesc(9), outm, descriptor(9))
END SUBROUTINE
