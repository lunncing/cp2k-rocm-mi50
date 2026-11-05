! SPDX-License-Identifier: GPL-2.0-or-later
#:set kind = 'r3d'
#:set kind2 = 'r3d'
#:set space = 'rs'
#:set type2type = lambda v,a,b: v
#define MARK_USED(value)
#define CPASSERT(condition) IF (.NOT. (condition)) ERROR STOP 'CPASSERT'
#define CPABORT(message) ERROR STOP message
MODULE fixture
   USE ISO_C_BINDING
   IMPLICIT NONE
   INTEGER, PARAMETER :: dp = C_DOUBLE, PW_MODE_LOCAL = 1
   INTEGER, PARAMETER :: pw_gpu_res_density_id = 1, pw_gpu_res_pot_id = 4
   TYPE group_type
   CONTAINS
      PROCEDURE :: sum => group_sum
   END TYPE
   TYPE para_type
      INTEGER :: mode = PW_MODE_LOCAL
      TYPE(group_type) :: group
   END TYPE
   TYPE pw_grid_type
      TYPE(para_type) :: para
      INTEGER :: id_nr = 1, npts(3) = [3, 2, 2], ngpts_cut_local = 12
      LOGICAL :: spherical = .FALSE.
      REAL(KIND=dp) :: dvol = 0.125_dp
      INTEGER, POINTER :: g_hatmap(:, :) => NULL()
      REAL(KIND=dp), POINTER :: gsq(:) => NULL()
   END TYPE
   TYPE pw_r3d_rs_type
      TYPE(pw_grid_type), POINTER :: pw_grid => NULL()
      REAL(KIND=dp), POINTER :: array(:, :, :) => NULL()
   END TYPE
   TYPE pw_c1d_gs_type
      TYPE(pw_grid_type), POINTER :: pw_grid => NULL()
      COMPLEX(KIND=dp), POINTER :: array(:) => NULL()
   END TYPE
   INTERFACE pw_gpu_res_ensure_host
      MODULE PROCEDURE pw_gpu_res_ensure_host_r, pw_gpu_res_ensure_host_g
   END INTERFACE
   INTERFACE pw_integrate_function
      MODULE PROCEDURE pw_integrate_function_r3d_rs
   END INTERFACE
   INTERFACE pw_zero
      MODULE PROCEDURE pw_zero_r3d_rs
   END INTERFACE
   INTERFACE
      SUBROUTINE res_fixture_begin(rho, g, n) BIND(C)
         IMPORT :: C_PTR, C_SIZE_T
         TYPE(C_PTR), VALUE :: rho, g
         INTEGER(C_SIZE_T), VALUE :: n
      END SUBROUTINE
      SUBROUTINE res_fixture_vxc(vxc, n) BIND(C)
         IMPORT :: C_PTR, C_SIZE_T
         TYPE(C_PTR), VALUE :: vxc
         INTEGER(C_SIZE_T), VALUE :: n
      END SUBROUTINE
      SUBROUTINE res_fixture_end() BIND(C)
      END SUBROUTINE
   END INTERFACE
CONTAINS
   #:include "res_bindings.inc"
   #:include "host_integrate.inc"
   #:include "host_copy.inc"
   #:include "host_axpy.inc"
   #:include "host_zero.inc"
   SUBROUTINE charge(rho, total_rho)
      TYPE(pw_r3d_rs_type), INTENT(IN) :: rho
      REAL(KIND=dp), OPTIONAL, INTENT(OUT) :: total_rho
      #:include "charge.inc"
   END SUBROUTINE
   REAL(KIND=dp) FUNCTION accurate_sum(a)
      REAL(KIND=dp), INTENT(IN) :: a(:, :, :)
      accurate_sum = SUM(a)
   END FUNCTION
   LOGICAL FUNCTION pw_compatible(a, b)
      TYPE(pw_grid_type), INTENT(IN) :: a, b
      pw_compatible = ALL(a%npts == b%npts)
   END FUNCTION
   SUBROUTINE group_sum(self, value)
      CLASS(group_type), INTENT(INOUT) :: self
      REAL(KIND=dp), INTENT(INOUT) :: value
      value = 2*value
   END SUBROUTINE
   SUBROUTINE timeset(name, handle)
      CHARACTER(LEN=*), INTENT(IN) :: name
      INTEGER, INTENT(OUT) :: handle
      handle = LEN(name)
   END SUBROUTINE
   SUBROUTINE timestop(handle)
      INTEGER, INTENT(IN) :: handle
   END SUBROUTINE
END MODULE

PROGRAM test_res_bindings
   USE fixture
   IMPLICIT NONE
   TYPE(pw_grid_type), POINTER :: grid
   TYPE(pw_r3d_rs_type) :: rho, copy, vh
   TYPE(pw_c1d_gs_type) :: g
   REAL(KIND=dp) :: expected(3, 2, 2), total
   LOGICAL :: enabled, done
   INTEGER :: i
   ALLOCATE (grid, rho%array(3, 2, 2), copy%array(3, 2, 2), vh%array(3, 2, 2), g%array(12))
   ALLOCATE (grid%gsq(12), grid%g_hatmap(12, 1))
   grid%gsq = 1
   grid%g_hatmap = 0
   rho%pw_grid => grid
   copy%pw_grid => grid
   vh%pw_grid => grid
   g%pw_grid => grid
   expected = RESHAPE([(REAL(i, dp), i=1, 12)], [3, 2, 2])
   rho%array = expected
   g%array = (2.0_dp, 3.0_dp)
   copy%array = -1
   CALL res_fixture_begin(C_LOC(rho%array), C_LOC(g%array), 12_C_SIZE_T)
   enabled = pw_gpu_res_enabled(pw_gpu_res_density_id)
   total = 123
   IF (pw_gpu_res_integrate_r(copy, total)) ERROR STOP 'wrong density accepted'
   IF (total /= 123) ERROR STOP 'fallback overwrote total'
   grid%para%mode = 2
   IF (pw_gpu_res_integrate_r(rho, total)) ERROR STOP 'distributed density accepted'
   grid%para%mode = PW_MODE_LOCAL
   CALL charge(rho) ! Optional total must not force synchronization.
   CALL charge(rho, total)
   IF (ABS(total + SUM(expected)*grid%dvol) > 1e-12_dp) ERROR STOP 'electron charge sign/scale'
   IF (enabled .AND. ANY(rho%array /= 0)) ERROR STOP 'charge downloaded density'
   CALL pw_copy_r3d_r3d_rs(rho, copy)
   IF (ANY(copy%array /= expected)) ERROR STOP 'copy read stale density'
   CALL pw_gpu_res_ensure_host(g)
   IF (ANY(g%array /= CMPLX(2.0_dp, 3.0_dp, dp))) ERROR STOP 'complex sync'
   CALL res_fixture_end()
   rho%array = expected
   CALL res_fixture_begin(C_LOC(rho%array), C_LOC(g%array), 12_C_SIZE_T)
   CALL pw_axpy_r3d_r3d_rs(rho, copy, 2.0_dp, 0.5_dp)
   IF (ANY(copy%array /= 2.5_dp*expected)) ERROR STOP 'AXPY read stale density'
   grid%para%mode = 2
   CALL charge(rho, total)
   IF (ABS(total + 2*SUM(expected)*grid%dvol) > 1e-12_dp) ERROR STOP 'distributed fallback'
   grid%para%mode = PW_MODE_LOCAL
   vh%array = 7
   CALL res_fixture_vxc(C_LOC(rho%array), 12_C_SIZE_T)
   done = pw_gpu_res_vtot_prepare(vh, rho, grid%dvol)
   IF (done .NEQV. enabled) ERROR STOP 'vtot preparation'
   IF (done) done = pw_gpu_res_vtot_skip_host()
   IF (done .NEQV. enabled) ERROR STOP 'vtot skip'
   IF (.NOT. done) CALL pw_axpy_r3d_r3d_rs(vh, rho, 1.0_dp, grid%dvol)
   CALL pw_gpu_res_ensure_host(rho)
   IF (ANY(rho%array /= 7 + grid%dvol*expected)) ERROR STOP 'vtot host synchronization'
   CALL res_fixture_end()
   DEALLOCATE (grid%gsq, grid%g_hatmap)
   DEALLOCATE (grid, rho%array, copy%array, vh%array, g%array)
   PRINT *, 'PASS residency Fortran bindings'
END PROGRAM
