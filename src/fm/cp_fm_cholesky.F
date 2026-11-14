!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief various cholesky decomposition related routines
!> \par History
!>      09.2002 created [fawzi]
!> \author Fawzi Mohamed
! **************************************************************************************************
MODULE cp_fm_cholesky
   USE cp_dlaf_utils_api,               ONLY: cp_dlaf_create_grid,&
                                              cp_dlaf_initialize
   USE cp_fm_dlaf_api,                  ONLY: cp_pdpotrf_dlaf,&
                                              cp_pdpotri_dlaf
   USE cp_fm_types,                     ONLY: cp_fm_type
   USE cp_log_handling,                 ONLY: cp_to_string
   USE ISO_C_BINDING,                   ONLY: C_INT, C_LOC, C_PTR
   USE kinds,                           ONLY: dp
!$ USE omp_lib,                         ONLY: omp_get_max_threads, omp_get_num_threads
#include "../base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   LOGICAL, PRIVATE, PARAMETER :: debug_this_module = .TRUE.
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'cp_fm_cholesky'

   PUBLIC :: cp_fm_cholesky_decompose, cp_fm_cholesky_invert, &
             cp_fm_cholesky_reduce, cp_fm_cholesky_restore, &
             cp_fm_cholesky_solve

   ! The following saved variables are Cholesky decomposition global
   ! Stores the default library for Cholesky decomposition
   INTEGER, SAVE, PUBLIC   :: cholesky_type = 0
   ! Minimum matrix size for the use of the DLAF Cholesky decomposition.
   ! ScaLAPACK is used as fallback for all smaller cases.
   INTEGER, SAVE, PUBLIC    :: dlaf_cholesky_n_min = 0
   ! Constants for the diag_type above
   INTEGER, PARAMETER, PUBLIC  :: FM_CHOLESKY_TYPE_SCALAPACK = 101, &
                                  FM_CHOLESKY_TYPE_DLAF = 104
   INTEGER, PARAMETER, PUBLIC :: FM_CHOLESKY_TYPE_DEFAULT = FM_CHOLESKY_TYPE_SCALAPACK

   INTERFACE
      FUNCTION pw_gpu_cholesky_fast_single_rank_c() BIND(C, name="pw_gpu_cholesky_fast_single_rank_c") RESULT(enabled)
         IMPORT :: C_INT
         INTEGER(C_INT) :: enabled
      END FUNCTION
      FUNCTION pw_gpu_openblas_threads_begin_c(target_threads) BIND(C, name="pw_gpu_openblas_threads_begin_c") RESULT(previous)
         IMPORT :: C_INT
         INTEGER(C_INT), VALUE :: target_threads
         INTEGER(C_INT) :: previous
      END FUNCTION
      SUBROUTINE pw_gpu_openblas_threads_end_c(previous) BIND(C, name="pw_gpu_openblas_threads_end_c")
         IMPORT :: C_INT
         INTEGER(C_INT), VALUE :: previous
      END SUBROUTINE
      FUNCTION pw_gpu_cholesky_restore_c(op_solve, side_right, trans_t, m, n, b, ldb, a, lda, outm, ldout) &
         BIND(C, name="pw_gpu_cholesky_restore_c") RESULT(done)
         IMPORT :: C_INT, C_PTR
         INTEGER(C_INT), VALUE :: op_solve, side_right, trans_t, m, n, ldb, lda, ldout
         TYPE(C_PTR), VALUE :: b, a, outm
         INTEGER(C_INT) :: done
      END FUNCTION
      FUNCTION pw_gpu_cholesky_reduce_c(n, a, lda, b, ldb) BIND(C, name="pw_gpu_cholesky_reduce_c") RESULT(done)
         IMPORT :: C_INT, C_PTR
         INTEGER(C_INT), VALUE :: n, lda, ldb
         TYPE(C_PTR), VALUE :: a, b
         INTEGER(C_INT) :: done
      END FUNCTION
   END INTERFACE

!***
CONTAINS

#if defined(__parallel)
   LOGICAL FUNCTION cholesky_single_rank(matrix, nrow, ncol) RESULT(single_rank)
      TYPE(cp_fm_type), INTENT(IN) :: matrix
      INTEGER, INTENT(IN) :: nrow, ncol
      INTEGER :: nthreads_cur

      nthreads_cur = 1
!$    nthreads_cur = omp_get_num_threads()
      single_rank = matrix%matrix_struct%para_env%num_pe == 1 .AND. &
                    ALL(matrix%matrix_struct%context%num_pe == 1) .AND. &
                    pw_gpu_cholesky_fast_single_rank_c() /= 0 .AND. nthreads_cur == 1 .AND. &
                    SIZE(matrix%local_data, 1) >= nrow .AND. SIZE(matrix%local_data, 2) >= ncol .AND. &
                    IS_CONTIGUOUS(matrix%local_data)
   END FUNCTION cholesky_single_rank

   FUNCTION cholesky_threads_begin() RESULT(previous)
      INTEGER(C_INT) :: previous
      INTEGER :: nthreads_max

      nthreads_max = 1
!$    nthreads_max = omp_get_max_threads()
      previous = pw_gpu_openblas_threads_begin_c(INT(nthreads_max, C_INT))
   END FUNCTION cholesky_threads_begin
#endif

! **************************************************************************************************
!> \brief used to replace a symmetric positive def. matrix M with its cholesky
!>      decomposition U: M = U^T * U, with U upper triangular
!> \param matrix the matrix to replace with its cholesky decomposition
!> \param n the number of row (and columns) of the matrix &
!>        (defaults to the min(size(matrix)))
!> \param info_out ...
!> \par History
!>      05.2002 created [JVdV]
!>      12.2002 updated, added n optional parm [fawzi]
!> \author Joost
! **************************************************************************************************
   SUBROUTINE cp_fm_cholesky_decompose(matrix, n, info_out)
      TYPE(cp_fm_type), INTENT(IN)             :: matrix
      INTEGER, INTENT(in), OPTIONAL            :: n
      INTEGER, INTENT(out), OPTIONAL           :: info_out

      CHARACTER(len=*), PARAMETER :: routineN = 'cp_fm_cholesky_decompose'

      INTEGER                                  :: handle, info, my_n
      REAL(KIND=dp), DIMENSION(:, :), POINTER  :: a
#if defined(__parallel)
      INTEGER, DIMENSION(9)                    :: desca
      INTEGER(C_INT)                           :: previous_threads
#endif

      CALL timeset(routineN, handle)

      my_n = MIN(matrix%matrix_struct%nrow_global, &
                 matrix%matrix_struct%ncol_global)
      IF (PRESENT(n)) THEN
         CPASSERT(n <= my_n)
         my_n = n
      END IF

      a => matrix%local_data

#if defined(__parallel)
      desca(:) = matrix%matrix_struct%descriptor(:)
      IF (cholesky_single_rank(matrix, my_n, my_n) .AND. cholesky_type /= FM_CHOLESKY_TYPE_DLAF) THEN
         previous_threads = cholesky_threads_begin()
         CALL dpotrf('U', my_n, a(1, 1), SIZE(a, 1), info)
         CALL pw_gpu_openblas_threads_end_c(previous_threads)
#if defined(__DLAF)
      ELSE IF (cholesky_type == FM_CHOLESKY_TYPE_DLAF .AND. matrix%matrix_struct%nrow_global >= dlaf_cholesky_n_min) THEN
         ! Initialize DLA-Future on-demand; if already initialized, does nothing
         CALL cp_dlaf_initialize()

         ! Create DLAF grid from BLACS context; if already present, does nothing
         CALL cp_dlaf_create_grid(matrix%matrix_struct%context%get_handle())

         CALL cp_pdpotrf_dlaf('U', my_n, a(:, :), 1, 1, desca, info)
#endif
      ELSE
         CALL pdpotrf('U', my_n, a(1, 1), 1, 1, desca, info)
      END IF
#else
      CALL dpotrf('U', my_n, a(1, 1), SIZE(a, 1), info)
#endif

      IF (PRESENT(info_out)) THEN
         info_out = info
      ELSE IF (info /= 0) THEN
         CALL cp_abort(__LOCATION__, &
                       "Cholesky decompose failed: the matrix is not positive definite or ill-conditioned.")
      END IF

      CALL timestop(handle)

   END SUBROUTINE cp_fm_cholesky_decompose

! **************************************************************************************************
!> \brief solves A*X = B for X, given the Cholesky decomposition U of the
!>      symmetric positive def. matrix A (A = U^T*U, U upper triangular),
!>      as produced by cp_fm_cholesky_decompose
!> \param matrix the Cholesky factor U of A, as produced by cp_fm_cholesky_decompose
!> \param matrixb on input the right-hand side(s) B, on output the solution(s) X
!> \param n the number of rows (and columns) of matrix (defaults to the min(size(matrix)))
!> \param info_out ...
! **************************************************************************************************
   SUBROUTINE cp_fm_cholesky_solve(matrix, matrixb, n, info_out)
      TYPE(cp_fm_type), INTENT(IN)             :: matrix
      TYPE(cp_fm_type), INTENT(INOUT)          :: matrixb
      INTEGER, INTENT(IN), OPTIONAL            :: n
      INTEGER, INTENT(OUT), OPTIONAL           :: info_out

      CHARACTER(len=*), PARAMETER :: routineN = 'cp_fm_cholesky_solve'

      INTEGER                                  :: handle, info, my_n, nrhs
      REAL(KIND=dp), DIMENSION(:, :), POINTER  :: a, b
#if defined(__parallel)
      INTEGER, DIMENSION(9)                    :: desca, descb
      INTEGER(C_INT)                           :: previous_threads
#endif

      CALL timeset(routineN, handle)

      my_n = MIN(matrix%matrix_struct%nrow_global, &
                 matrix%matrix_struct%ncol_global)
      IF (PRESENT(n)) THEN
         CPASSERT(n <= my_n)
         my_n = n
      END IF
      nrhs = matrixb%matrix_struct%ncol_global

      a => matrix%local_data
      b => matrixb%local_data

#if defined(__parallel)
      desca(:) = matrix%matrix_struct%descriptor(:)
      descb(:) = matrixb%matrix_struct%descriptor(:)
      IF (cholesky_single_rank(matrix, my_n, my_n) .AND. cholesky_single_rank(matrixb, my_n, nrhs)) THEN
         previous_threads = cholesky_threads_begin()
         CALL dpotrs('U', my_n, nrhs, a(1, 1), SIZE(a, 1), b(1, 1), SIZE(b, 1), info)
         CALL pw_gpu_openblas_threads_end_c(previous_threads)
      ELSE
         CALL pdpotrs('U', my_n, nrhs, a(1, 1), 1, 1, desca, b(1, 1), 1, 1, descb, info)
      END IF
#else
      CALL dpotrs('U', my_n, nrhs, a(1, 1), SIZE(a, 1), b(1, 1), SIZE(b, 1), info)
#endif

      IF (PRESENT(info_out)) THEN
         info_out = info
      ELSE IF (info /= 0) THEN
         CALL cp_abort(__LOCATION__, &
                       "Cholesky solve failed: "// &
                       "the matrix is not positive definite or ill-conditioned, info="//cp_to_string(info))
      END IF

      CALL timestop(handle)

   END SUBROUTINE cp_fm_cholesky_solve

! **************************************************************************************************
!> \brief used to replace the cholesky decomposition by the inverse
!> \param matrix the matrix to invert (must be an upper triangular matrix)
!> \param n size of the matrix to invert (defaults to the min(size(matrix)))
!> \param info_out ...
!> \par History
!>      05.2002 created [JVdV]
!> \author Joost VandeVondele
! **************************************************************************************************
   SUBROUTINE cp_fm_cholesky_invert(matrix, n, info_out)
      TYPE(cp_fm_type), INTENT(IN)              :: matrix
      INTEGER, INTENT(in), OPTIONAL             :: n
      INTEGER, INTENT(OUT), OPTIONAL            :: info_out

      CHARACTER(len=*), PARAMETER :: routineN = 'cp_fm_cholesky_invert'
      REAL(KIND=dp), DIMENSION(:, :), POINTER   :: a
      INTEGER                                   :: info, handle, my_n
#if defined(__parallel)
      INTEGER, DIMENSION(9)                     :: desca
      INTEGER(C_INT)                            :: previous_threads
#endif

      CALL timeset(routineN, handle)

      my_n = MIN(matrix%matrix_struct%nrow_global, &
                 matrix%matrix_struct%ncol_global)
      IF (PRESENT(n)) THEN
         CPASSERT(n <= my_n)
         my_n = n
      END IF

      a => matrix%local_data

#if defined(__parallel)

      desca(:) = matrix%matrix_struct%descriptor(:)

      IF (cholesky_single_rank(matrix, my_n, my_n) .AND. cholesky_type /= FM_CHOLESKY_TYPE_DLAF) THEN
         previous_threads = cholesky_threads_begin()
         CALL dpotri('U', my_n, a(1, 1), SIZE(a, 1), info)
         CALL pw_gpu_openblas_threads_end_c(previous_threads)
#if defined(__DLAF)
      ELSE IF (cholesky_type == FM_CHOLESKY_TYPE_DLAF .AND. matrix%matrix_struct%nrow_global >= dlaf_cholesky_n_min) THEN
         ! Initialize DLA-Future on-demand; if already initialized, does nothing
         CALL cp_dlaf_initialize()

         ! Create DLAF grid from BLACS context; if already present, does nothing
         CALL cp_dlaf_create_grid(matrix%matrix_struct%context%get_handle())

         CALL cp_pdpotri_dlaf('U', my_n, a(:, :), 1, 1, desca, info)
#endif
      ELSE
         CALL pdpotri('U', my_n, a(1, 1), 1, 1, desca, info)
      END IF

#else

      CALL dpotri('U', my_n, a(1, 1), SIZE(a, 1), info)

#endif

      IF (PRESENT(info_out)) THEN
         info_out = info
      ELSE
         IF (info /= 0) &
            CPABORT("Cholesky invert failed: the matrix is not positive definite or ill-conditioned.")
      END IF

      CALL timestop(handle)

   END SUBROUTINE cp_fm_cholesky_invert

! **************************************************************************************************
!> \brief reduce a matrix pencil A,B to normal form
!>      B has to be cholesky decomposed with  cp_fm_cholesky_decompose
!>      before calling this routine
!>      A,B -> inv(U^T)*A*inv(U),1
!>      (AX=BX -> inv(U^T)*A*inv(U)*U*X=U*X hence evecs U*X)
!> \param matrix the symmetric matrix A
!> \param matrixb the cholesky decomposition of matrix B
!> \param itype ...
!> \par History
!>      05.2002 created [JVdV]
!> \author Joost VandeVondele
! **************************************************************************************************
   SUBROUTINE cp_fm_cholesky_reduce(matrix, matrixb, itype)
      TYPE(cp_fm_type), INTENT(IN)              :: matrix, matrixb
      INTEGER, OPTIONAL                         :: itype

      CHARACTER(len=*), PARAMETER :: routineN = 'cp_fm_cholesky_reduce'
      REAL(KIND=dp), DIMENSION(:, :), POINTER   :: a, b
      INTEGER                                   :: info, handle
      INTEGER                                   :: n, my_itype
#if defined(__parallel)
      REAL(KIND=dp)                             :: scale
      INTEGER, DIMENSION(9)                     :: desca, descb
      INTEGER                                   :: row, column
      INTEGER(C_INT)                            :: previous_threads, gpu_done
      EXTERNAL                                  :: dtrsm
#endif

      CALL timeset(routineN, handle)

      n = matrix%matrix_struct%nrow_global

      my_itype = 1
      IF (PRESENT(itype)) my_itype = itype

      a => matrix%local_data
      b => matrixb%local_data

#if defined(__parallel)

      desca(:) = matrix%matrix_struct%descriptor(:)
      descb(:) = matrixb%matrix_struct%descriptor(:)

      IF (cholesky_single_rank(matrix, n, n) .AND. cholesky_single_rank(matrixb, n, n)) THEN
         info = 0
         IF (my_itype == 1 .AND. n >= 512) THEN
            gpu_done = pw_gpu_cholesky_reduce_c(n, C_LOC(a(1, 1)), SIZE(a, 1), C_LOC(b(1, 1)), SIZE(b, 1))
            IF (gpu_done == 0) THEN
               DO column = 1, n
                  DO row = column + 1, n
                     a(row, column) = a(column, row)
                  END DO
               END DO
               previous_threads = cholesky_threads_begin()
               CALL dtrsm('L', 'U', 'T', 'N', n, n, 1.0_dp, b(1, 1), SIZE(b, 1), a(1, 1), SIZE(a, 1))
               CALL dtrsm('R', 'U', 'N', 'N', n, n, 1.0_dp, b(1, 1), SIZE(b, 1), a(1, 1), SIZE(a, 1))
               CALL pw_gpu_openblas_threads_end_c(previous_threads)
            END IF
         ELSE
            previous_threads = pw_gpu_openblas_threads_begin_c(1_C_INT)
            CALL dsygst(my_itype, 'U', n, a(1, 1), SIZE(a, 1), b(1, 1), SIZE(b, 1), info)
            CALL pw_gpu_openblas_threads_end_c(previous_threads)
         END IF
      ELSE
         CALL pdsygst(my_itype, 'U', n, a(1, 1), 1, 1, desca, b(1, 1), 1, 1, descb, scale, info)

      ! this is supposed to be one in current version of lapack
      ! if not, eigenvalues have to be scaled by this number
         IF (scale /= 1.0_dp) &
            CPABORT("scale not equal 1 (scale="//cp_to_string(scale)//")")
      END IF
#else

      CALL dsygst(my_itype, 'U', n, a(1, 1), n, b(1, 1), n, info)

#endif

      CPASSERT(info == 0)

      CALL timestop(handle)

   END SUBROUTINE cp_fm_cholesky_reduce

! **************************************************************************************************
!> \brief apply Cholesky decomposition
!>        op can be "SOLVE" (out = U^-1 * in) or "MULTIPLY"  (out = U * in)
!>        pos can be "LEFT" or "RIGHT" (U at the left or at the right)
!> \param fm_matrix ...
!> \param neig ...
!> \param fm_matrixb ...
!> \param fm_matrixout ...
!> \param op ...
!> \param pos ...
!> \param transa ...
! **************************************************************************************************
   SUBROUTINE cp_fm_cholesky_restore(fm_matrix, neig, fm_matrixb, fm_matrixout, op, pos, transa)
      TYPE(cp_fm_type), INTENT(IN)              :: fm_matrix, fm_matrixb, fm_matrixout
      INTEGER, INTENT(IN)                       :: neig
      CHARACTER(LEN=*), INTENT(IN)              :: op
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL    :: pos
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL    :: transa

      CHARACTER(len=*), PARAMETER :: routineN = 'cp_fm_cholesky_restore'
      REAL(KIND=dp), DIMENSION(:, :), POINTER   :: a, b, outm
      REAL(KIND=dp)                             :: alpha
      INTEGER                                   :: handle, mdim, n
      CHARACTER                                 :: chol_pos, chol_transa
      EXTERNAL                                  :: dtrsm, dtrmm
#if defined(__parallel)
      INTEGER                                   :: i, bdim
      INTEGER(C_INT)                            :: previous_threads, gpu_done
      INTEGER, DIMENSION(9)                     :: desca, descb, descout
#elif defined(__MKL) && (2025 <= __INTEL_MKL__)
      REAL(KIND=dp)                             :: beta
#endif

      CALL timeset(routineN, handle)

      ! wrong argument op
      CPASSERT((op == "SOLVE") .OR. (op == "MULTIPLY"))

      chol_pos = 'L'
      IF (PRESENT(pos)) chol_pos = pos(1:1)
      CPASSERT((chol_pos == 'L') .OR. (chol_pos == 'R'))

      chol_transa = 'N'
      IF (PRESENT(transa)) chol_transa = transa

      ! notice b is the cholesky guy
      a => fm_matrix%local_data
      b => fm_matrixb%local_data
      outm => fm_matrixout%local_data
      n = fm_matrix%matrix_struct%nrow_global
      mdim = MERGE(1, 2, 'L' == chol_pos)
      alpha = 1.0_dp

#if defined(__parallel)
      desca(:) = fm_matrix%matrix_struct%descriptor(:)
      descb(:) = fm_matrixb%matrix_struct%descriptor(:)
      descout(:) = fm_matrixout%matrix_struct%descriptor(:)
      bdim = MERGE(neig, n, chol_pos == 'R')
      IF (cholesky_single_rank(fm_matrix, n, neig) .AND. &
          cholesky_single_rank(fm_matrixb, bdim, bdim) .AND. cholesky_single_rank(fm_matrixout, n, neig)) THEN
         gpu_done = pw_gpu_cholesky_restore_c(MERGE(1, 0, op == 'SOLVE'), MERGE(1, 0, chol_pos == 'R'), &
                                             MERGE(1, 0, chol_transa == 'T' .OR. chol_transa == 't' .OR. &
                                                   chol_transa == 'C' .OR. chol_transa == 'c'), &
                                             n, neig, C_LOC(b(1, 1)), SIZE(b, 1), C_LOC(a(1, 1)), SIZE(a, 1), &
                                             C_LOC(outm(1, 1)), SIZE(outm, 1))
         IF (gpu_done == 0) THEN
            outm(1:n, 1:neig) = a(1:n, 1:neig)
            previous_threads = cholesky_threads_begin()
            IF (op == "SOLVE") THEN
               CALL dtrsm(chol_pos, 'U', chol_transa, 'N', n, neig, alpha, b(1, 1), SIZE(b, 1), outm(1, 1), SIZE(outm, 1))
            ELSE
               CALL dtrmm(chol_pos, 'U', chol_transa, 'N', n, neig, alpha, b(1, 1), SIZE(b, 1), outm(1, 1), SIZE(outm, 1))
            END IF
            CALL pw_gpu_openblas_threads_end_c(previous_threads)
         END IF
      ELSE
         DO i = 1, neig
            CALL pdcopy(n, a(1, 1), 1, i, desca, 1, outm(1, 1), 1, i, descout, 1)
         END DO
         IF (op == "SOLVE") THEN
            CALL pdtrsm(chol_pos, 'U', chol_transa, 'N', n, neig, alpha, b(1, 1), 1, 1, descb, outm(1, 1), 1, 1, descout)
         ELSE
            CALL pdtrmm(chol_pos, 'U', chol_transa, 'N', n, neig, alpha, b(1, 1), 1, 1, descb, outm(1, 1), 1, 1, descout)
         END IF
      END IF
#elif defined(__MKL) && (2025 <= __INTEL_MKL__)
      beta = 0.0_dp
      IF (op == "SOLVE") THEN
         CALL dtrsm_oop(chol_pos, 'U', chol_transa, 'N', n, neig, alpha, b(1, 1), SIZE(b, mdim), a(1, 1), n, beta, outm(1, 1), n)
      ELSE
         CALL dtrmm_oop(chol_pos, 'U', chol_transa, 'N', n, neig, alpha, b(1, 1), SIZE(b, mdim), a(1, 1), n, beta, outm(1, 1), n)
      END IF
#else
      CALL dcopy(neig*n, a(1, 1), 1, outm(1, 1), 1)
      IF (op == "SOLVE") THEN
         CALL dtrsm(chol_pos, 'U', chol_transa, 'N', n, neig, alpha, b(1, 1), SIZE(b, mdim), outm(1, 1), n)
      ELSE
         CALL dtrmm(chol_pos, 'U', chol_transa, 'N', n, neig, alpha, b(1, 1), SIZE(b, mdim), outm(1, 1), n)
      END IF
#endif

      CALL timestop(handle)

   END SUBROUTINE cp_fm_cholesky_restore

END MODULE cp_fm_cholesky
