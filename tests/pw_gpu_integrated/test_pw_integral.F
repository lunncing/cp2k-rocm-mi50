! SPDX-License-Identifier: GPL-2.0-or-later
! The runner includes the actual integral template; only its CP2K dependencies
! are stubbed. BLAS and the Fortran/C interface are real.
#:set pw_kinds = ['r1d', 'c1d', 'r3d', 'c3d']
#:set pw_spaces = ['rs', 'gs']
#:set pairs = [('r1d', 'r1d'), ('r1d', 'c1d'), ('c1d', 'r1d'), ('c1d', 'c1d'), &
            & ('r3d', 'r3d'), ('r3d', 'c3d'), ('c3d', 'r3d'), ('c3d', 'c3d')]
#:set declaration = {'r': 'REAL(KIND=dp)', 'c': 'COMPLEX(KIND=dp)'}
#:set dims = {'1': ':', '3': ':, :, :'}
#:set shape = {'1': '257', '3': '7, 3, 5'}
#define CPABORT(message) ERROR STOP message
MODULE integral_fixture
   USE ISO_C_BINDING, ONLY: C_DOUBLE
   IMPLICIT NONE
   INTEGER, PARAMETER :: dp = C_DOUBLE, HALFSPACE = 1, PW_MODE_DISTRIBUTED = 2
   INTEGER, PARAMETER :: do_accurate_sum = 0, do_standard_sum = 1
   INTEGER :: accurate_calls = 0
   TYPE group_type
      INTEGER :: calls = 0
   CONTAINS
      PROCEDURE :: sum => group_sum
   END TYPE
   TYPE parallel_type
      INTEGER :: mode = 0
      TYPE(group_type) :: group
   END TYPE
   TYPE grid_type
      REAL(KIND=dp) :: vol = 2.5_dp, dvol = 0.125_dp
      INTEGER :: grid_span = 0
      LOGICAL :: have_g0 = .FALSE.
      TYPE(parallel_type) :: para
   END TYPE
   #:for space in pw_spaces
      #:for kind in pw_kinds
         TYPE pw_${kind}$_${space}$_type
            ${declaration[kind[0]]}$, POINTER :: array(${dims[kind[1]]}$) => NULL()
            TYPE(grid_type), POINTER :: pw_grid => NULL()
         END TYPE
      #:endfor
   #:endfor
   INTERFACE accurate_sum
      MODULE PROCEDURE accurate_sum_1, accurate_sum_3
   END INTERFACE
   INTERFACE accurate_dot_product
      MODULE PROCEDURE accurate_dot_1, accurate_dot_3
   END INTERFACE
CONTAINS
   SUBROUTINE group_sum(self, value)
      CLASS(group_type), INTENT(INOUT) :: self
      REAL(KIND=dp), INTENT(INOUT) :: value
      self%calls = self%calls + 1
      value = 3.0_dp*value
   END SUBROUTINE

   SUBROUTINE timeset(name, handle)
      CHARACTER(LEN=*), INTENT(IN) :: name
      INTEGER, INTENT(OUT) :: handle
      handle = LEN_TRIM(name)
   END SUBROUTINE

   SUBROUTINE timestop(handle)
      INTEGER, INTENT(IN) :: handle
      IF (handle <= 0) ERROR STOP 'Invalid timer'
   END SUBROUTINE

   #:for rank in ['1', '3']
      FUNCTION accurate_sum_${rank}$ (a) RESULT(value)
         REAL(KIND=dp), INTENT(IN) :: a(${dims[rank]}$)
         REAL(KIND=dp) :: value
         accurate_calls = accurate_calls + 1
         value = SUM(a)
      END FUNCTION

      FUNCTION accurate_dot_${rank}$ (a, b) RESULT(value)
         REAL(KIND=dp), INTENT(IN) :: a(${dims[rank]}$), b(${dims[rank]}$)
         REAL(KIND=dp) :: value
         accurate_calls = accurate_calls + 1
         value = SUM(a*b)
      END FUNCTION
   #:endfor

   #:for kind, kind2 in pairs
      #:for space in pw_spaces
         #:include "pw_integral_ab.inc"
      #:endfor
   #:endfor
END MODULE

PROGRAM test_pw_integral
   USE integral_fixture
   IMPLICIT NONE
   INTEGER, ALLOCATABLE :: seed(:)
   INTEGER :: nseed, i
   CHARACTER(LEN=32) :: argument
   CALL GET_COMMAND_ARGUMENT(1, argument)
   IF (argument == 'incompatible') CALL incompatible_grids()
   CALL RANDOM_SEED(SIZE=nseed)
   ALLOCATE (seed(nseed))
   seed = [(17321 + 37*i, i=1, nseed)]
   CALL RANDOM_SEED(PUT=seed)
   DEALLOCATE (seed)
   #:for kind, kind2 in pairs
      #:for space in pw_spaces
         CALL check_${kind}$_${kind2}$_${space}$ ()
      #:endfor
   #:endfor
   CALL random_lengths()
   CALL strided_arrays()
   PRINT '(a)', 'PASS Fortran random vectors, 16 variants, sum modes, scaling, half-space, and reductions'
CONTAINS
   SUBROUTINE close(actual, expected, magnitude)
      REAL(KIND=dp), INTENT(IN) :: actual, expected, magnitude
      IF (.NOT. (ABS(actual - expected) <= 128*EPSILON(1.0_dp)*MAX(1.0_dp, magnitude))) THEN
         PRINT *, 'Integral mismatch:', actual, expected
         ERROR STOP 'Numerical mismatch'
      END IF
   END SUBROUTINE

   #:for kind, kind2 in pairs
      #:for space in pw_spaces
         SUBROUTINE check_${kind}$_${kind2}$_${space}$ ()
            TYPE(pw_${kind}$_${space}$_type) :: a
            TYPE(pw_${kind2}$_${space}$_type) :: b
            TYPE(grid_type), TARGET :: grid
            REAL(KIND=dp) :: r(${shape[kind[1]]}$), s(${shape[kind[1]]}$)
            REAL(KIND=dp) :: products(${shape[kind[1]]}$), base, expected, actual, magnitude
            INTEGER :: mode, just, local, distributed, span, g0, expected_calls
            ALLOCATE (a%array(${shape[kind[1]]}$), b%array(${shape[kind[1]]}$))
            a%pw_grid => grid
            b%pw_grid => grid
            #:for name, dtype in [('a', kind), ('b', kind2)]
               CALL RANDOM_NUMBER(r)
               CALL RANDOM_NUMBER(s)
               #:if dtype[0] == 'c'
                  ${name}$%array = CMPLX(2*r - 1, 2*s - 1, KIND=dp)
               #:else
                  ${name}$%array = 2*r - 1
               #:endif
            #:endfor
            #:if kind[0] == 'c'
               products = REAL(CONJG(a%array)*b%array, KIND=dp)
            #:else
               products = a%array*REAL(b%array, KIND=dp)
            #:endif
            base = SUM(products)
            magnitude = 16*SUM(ABS(a%array)*ABS(b%array))
            DO mode = -1, 1 ! Omitted, accurate, standard.
               DO just = 0, 1
                  DO local = 0, 1
                     DO distributed = 0, 1
                        DO span = 0, 1
                           DO g0 = 0, 1
                              grid%grid_span = span
                              grid%have_g0 = g0 == 1
                              grid%para%mode = distributed*PW_MODE_DISTRIBUTED
                              grid%para%group%calls = 0
                              accurate_calls = 0
                              expected = base
                              #:if space == 'gs'
                                 IF (just == 0) expected = expected*grid%vol
                              #:else
                                 IF (just == 0) expected = expected*grid%dvol
                              #:endif
                              #:if kind[1] == '1'
                                 IF (span == HALFSPACE) THEN
                                    expected = 2*expected
                                    IF (g0 == 1) expected = expected - products(1)
                                 END IF
                              #:endif
                              expected_calls = 0
                              IF (distributed == 1 .AND. local == 0) THEN
                                 expected = 3*expected
                                 expected_calls = 1
                              END IF
                              IF (mode == -1) THEN
                                 actual = pw_integral_ab_${kind}$_${kind2}$_${space}$ (&
                                    a, b, just_sum=just == 1, local_only=local == 1)
                              ELSE
                                 actual = pw_integral_ab_${kind}$_${kind2}$_${space}$ (&
                                    a, b, sumtype=mode, just_sum=just == 1, local_only=local == 1)
                              END IF
                              CALL close(actual, expected, magnitude)
                              IF (grid%para%group%calls /= expected_calls) ERROR STOP 'Wrong reduction'
                              #:if kind == 'c1d' and kind2 == 'c1d' and space == 'gs'
                                 IF (accurate_calls /= 0) ERROR STOP 'Contiguous complex dot used scalar sum'
                              #:else
                                 IF (accurate_calls /= MERGE(0, 1, mode == 1)) ERROR STOP 'Changed sum mode'
                              #:endif
                           END DO
                        END DO
                     END DO
                  END DO
               END DO
            END DO
            DEALLOCATE (a%array, b%array)
         END SUBROUTINE
      #:endfor
   #:endfor

   SUBROUTINE random_lengths()
      TYPE(pw_c1d_gs_type) :: a, b
      TYPE(grid_type), TARGET :: grid
      INTEGER, PARAMETER :: sizes(*) = [0, 1, 2, 3, 7, 31, 64, 257, 1023, 4096, 65537]
      REAL(KIND=dp), ALLOCATABLE :: r(:), s(:)
      REAL(KIND=dp) :: expected, magnitude, actual
      INTEGER :: n, j, trial, mode
      a%pw_grid => grid
      b%pw_grid => grid
      DO j = 1, SIZE(sizes)
         n = sizes(j)
         ALLOCATE (a%array(n), b%array(n), r(n), s(n))
         DO trial = 1, 8
            CALL RANDOM_NUMBER(r)
            CALL RANDOM_NUMBER(s)
            a%array = CMPLX(2*r - 1, 2*s - 1, KIND=dp)
            CALL RANDOM_NUMBER(r)
            CALL RANDOM_NUMBER(s)
            b%array = CMPLX(2*r - 1, 2*s - 1, KIND=dp)
            expected = SUM(REAL(CONJG(a%array)*b%array, KIND=dp))
            magnitude = SUM(ABS(a%array)*ABS(b%array))
            DO mode = 0, 1
               accurate_calls = 0
               actual = pw_integral_ab_c1d_c1d_gs(a, b, sumtype=mode, just_sum=.TRUE.)
               CALL close(actual, expected, magnitude)
               IF (accurate_calls /= 0) ERROR STOP 'Random complex dot used scalar sum'
            END DO
         END DO
         ! Empty local grids must still participate in a distributed reduction.
         IF (n == 0) THEN
            grid%grid_span = HALFSPACE
            grid%para%mode = PW_MODE_DISTRIBUTED
            actual = pw_integral_ab_c1d_c1d_gs(a, b)
            CALL close(actual, 0.0_dp, 1.0_dp)
            IF (grid%para%group%calls /= 1) ERROR STOP 'Missing empty-grid reduction'
            grid%grid_span = 0
            grid%para%mode = 0
         END IF
         DEALLOCATE (a%array, b%array, r, s)
      END DO
   END SUBROUTINE

   SUBROUTINE strided_arrays()
      TYPE(pw_c1d_gs_type) :: a, b
      TYPE(grid_type), TARGET :: grid
      COMPLEX(KIND=dp), TARGET :: storage_a(34), storage_b(34)
      REAL(KIND=dp) :: r(34), s(34), actual, expected, magnitude
      INTEGER :: stride_case, mode
      CALL RANDOM_NUMBER(r)
      CALL RANDOM_NUMBER(s)
      storage_a = CMPLX(2*r - 1, 2*s - 1, KIND=dp)
      CALL RANDOM_NUMBER(r)
      CALL RANDOM_NUMBER(s)
      storage_b = CMPLX(2*r - 1, 2*s - 1, KIND=dp)
      a%pw_grid => grid
      b%pw_grid => grid
      DO stride_case = 1, 3
         a%array => storage_a(1:17)
         b%array => storage_b(1:17)
         IF (stride_case /= 2) a%array => storage_a(2:34:2)
         IF (stride_case /= 1) b%array => storage_b(34:2:-2)
         expected = SUM(REAL(CONJG(a%array)*b%array, KIND=dp))
         magnitude = SUM(ABS(a%array)*ABS(b%array))
         DO mode = 0, 1
            accurate_calls = 0
            actual = pw_integral_ab_c1d_c1d_gs(a, b, sumtype=mode, just_sum=.TRUE.)
            CALL close(actual, expected, magnitude)
            IF (accurate_calls /= MERGE(0, 1, mode == 1)) ERROR STOP 'Changed strided sum mode'
         END DO
      END DO
   END SUBROUTINE

   SUBROUTINE incompatible_grids()
      TYPE(pw_c1d_gs_type) :: a, b
      TYPE(grid_type), TARGET :: grid_a, grid_b
      REAL(KIND=dp) :: actual
      a%pw_grid => grid_a
      b%pw_grid => grid_b
      ALLOCATE (a%array(1), b%array(1))
      a%array = 1
      b%array = 1
      actual = pw_integral_ab_c1d_c1d_gs(a, b)
      PRINT *, actual
      ERROR STOP 'Accepted incompatible grids'
   END SUBROUTINE
END PROGRAM
