! SPDX-License-Identifier: GPL-2.0-or-later
#define MARK_USED(value)
#define CPASSERT(condition) IF (.NOT. (condition)) ERROR STOP 'CPASSERT'
#define __LOCATION__ 'fixture'
MODULE grid_fixture
   USE ISO_C_BINDING
   IMPLICIT NONE
   INTEGER, PARAMETER :: dp = C_DOUBLE, PW_MODE_LOCAL = 1
   INTEGER, PARAMETER :: pw_gpu_res_density_id = 1, pw_gpu_res_pot_id = 4
   INTEGER, PARAMETER :: pw_interp = 1, spline3_pbc_interp = 2
   INTEGER :: transfers = 0
   TYPE para_type
      INTEGER :: mode = PW_MODE_LOCAL
   END TYPE
   TYPE pw_grid_type
      TYPE(para_type) :: para
      INTEGER :: id_nr = 1, npts(3) = [3, 2, 2], ngpts_cut_local = 12, ngpts = 12, reference = 0
      INTEGER :: bounds(2, 3) = RESHAPE([0, 2, 0, 1, 0, 1], [2, 3])
      LOGICAL :: spherical = .FALSE.
      REAL(dp) :: dvol = 0.125_dp
      INTEGER, POINTER :: g_hatmap(:, :) => NULL(), gidx(:) => NULL()
      REAL(dp), POINTER :: gsq(:) => NULL()
   END TYPE
   TYPE pw_r3d_rs_type
      TYPE(pw_grid_type), POINTER :: pw_grid => NULL()
      REAL(dp), POINTER :: array(:, :, :) => NULL()
   END TYPE
   TYPE pw_c1d_gs_type
      TYPE(pw_grid_type), POINTER :: pw_grid => NULL()
      COMPLEX(dp), POINTER :: array(:) => NULL()
   END TYPE
   TYPE offload_buffer_type
      REAL(dp), CONTIGUOUS, POINTER :: host_buffer(:) => NULL()
      TYPE(C_PTR) :: c_ptr = C_NULL_PTR
   END TYPE
   TYPE realspace_grid_desc_type
      TYPE(pw_grid_type), POINTER :: pw => NULL()
      LOGICAL :: distributed = .FALSE., parallel = .FALSE.
      INTEGER :: border = 0
   END TYPE
   TYPE realspace_grid_type
      TYPE(realspace_grid_desc_type), POINTER :: desc => NULL()
      INTEGER :: ngpts_local = 12, npts_local(3) = [3, 2, 2]
      INTEGER :: lb_local(3) = 0, ub_local(3) = [2, 1, 1]
      TYPE(offload_buffer_type) :: buffer
   END TYPE
   TYPE realspace_grid_desc_p_type
      TYPE(realspace_grid_desc_type), POINTER :: rs_desc => NULL()
   END TYPE
   TYPE gridlevel_info_type
      INTEGER :: ngrid_levels = 1
   END TYPE
   TYPE pool_type
      TYPE(pw_grid_type), POINTER :: grid => NULL()
   END TYPE
   TYPE pw_pool_p_type
      TYPE(pool_type) :: pool
   END TYPE
   TYPE pw_env_type
      TYPE(gridlevel_info_type), POINTER :: gridlevel_info => NULL()
      TYPE(pw_pool_p_type), POINTER :: pools(:) => NULL()
      INTEGER :: interp_section = pw_interp
   END TYPE
   INTERFACE pw_gpu_res_ensure_host
      MODULE PROCEDURE pw_gpu_res_ensure_host_r, pw_gpu_res_ensure_host_g
   END INTERFACE
   INTERFACE pw_pools_create_pws
      MODULE PROCEDURE create_r, create_g
   END INTERFACE
   INTERFACE pw_pools_give_back_pws
      MODULE PROCEDURE free_r, free_g
   END INTERFACE
   INTERFACE pw_transfer
      MODULE PROCEDURE transfer_rg, transfer_gr
   END INTERFACE
   INTERFACE pw_copy
      MODULE PROCEDURE copy_r, copy_g
   END INTERFACE
   INTERFACE pw_zero
      MODULE PROCEDURE zero_r, zero_g
   END INTERFACE
   INTERFACE
      SUBROUTINE grid_fixture_begin() BIND(C)
      END SUBROUTINE
      SUBROUTINE grid_fixture_collocate(buffer) BIND(C)
         IMPORT C_PTR
         TYPE(C_PTR), VALUE :: buffer
      END SUBROUTINE
      SUBROUTINE grid_fixture_potential(host, n) BIND(C)
         IMPORT C_PTR, C_SIZE_T
         TYPE(C_PTR), VALUE :: host
         INTEGER(C_SIZE_T), VALUE :: n
      END SUBROUTINE
      SUBROUTINE grid_fixture_end() BIND(C)
      END SUBROUTINE
   END INTERFACE
CONTAINS
#include "grid_pw_bindings.inc"
#include "grid_rs_bindings.inc"
#include "grid_offload_bindings.inc"
   SUBROUTINE create_r(pools, pw)
      TYPE(pw_pool_p_type), INTENT(IN) :: pools(:)
      TYPE(pw_r3d_rs_type), ALLOCATABLE, INTENT(OUT) :: pw(:)
      INTEGER :: i
      ALLOCATE (pw(SIZE(pools)))
      DO i = 1, SIZE(pools)
         pw(i)%pw_grid => pools(i)%pool%grid
         ALLOCATE (pw(i)%array(3, 2, 2))
         pw(i)%array = -987
      END DO
   END SUBROUTINE
   SUBROUTINE create_g(pools, pw)
      TYPE(pw_pool_p_type), INTENT(IN) :: pools(:)
      TYPE(pw_c1d_gs_type), ALLOCATABLE, INTENT(OUT) :: pw(:)
      INTEGER :: i
      ALLOCATE (pw(SIZE(pools)))
      DO i = 1, SIZE(pools)
         pw(i)%pw_grid => pools(i)%pool%grid
         ALLOCATE (pw(i)%array(12))
      END DO
   END SUBROUTINE
   SUBROUTINE free_r(pools, pw)
      TYPE(pw_pool_p_type) :: pools(:)
      TYPE(pw_r3d_rs_type), ALLOCATABLE :: pw(:)
      INTEGER :: i
      DO i = 1, SIZE(pw)
         DEALLOCATE (pw(i)%array)
      END DO
      DEALLOCATE (pw)
   END SUBROUTINE
   SUBROUTINE free_g(pools, pw)
      TYPE(pw_pool_p_type) :: pools(:)
      TYPE(pw_c1d_gs_type), ALLOCATABLE :: pw(:)
      INTEGER :: i
      DO i = 1, SIZE(pw)
         DEALLOCATE (pw(i)%array)
      END DO
      DEALLOCATE (pw)
   END SUBROUTINE
   SUBROUTINE pw_env_get(env, rs_descs, pw_pools, gridlevel_info, auxbas_grid)
      TYPE(pw_env_type), INTENT(IN) :: env
      TYPE(realspace_grid_desc_p_type), POINTER, OPTIONAL :: rs_descs(:)
      TYPE(pw_pool_p_type), POINTER, OPTIONAL :: pw_pools(:)
      TYPE(gridlevel_info_type), POINTER, OPTIONAL :: gridlevel_info
      INTEGER, OPTIONAL :: auxbas_grid
      IF (PRESENT(rs_descs)) NULLIFY (rs_descs)
      IF (PRESENT(pw_pools)) pw_pools => env%pools
      IF (PRESENT(gridlevel_info)) gridlevel_info => env%gridlevel_info
      IF (PRESENT(auxbas_grid)) auxbas_grid = 1
   END SUBROUTINE
   SUBROUTINE section_vals_val_get(section, name, i_val)
      INTEGER, INTENT(IN) :: section
      CHARACTER(*) :: name
      INTEGER, INTENT(OUT) :: i_val
      i_val = section
   END SUBROUTINE
   SUBROUTINE transfer_rs2pw(rs, pw)
      TYPE(realspace_grid_type), INTENT(IN) :: rs
      TYPE(pw_r3d_rs_type), INTENT(INOUT) :: pw
      transfers = transfers + 1
      pw%array = RESHAPE(rs%buffer%host_buffer, SHAPE(pw%array))
   END SUBROUTINE
   SUBROUTINE transfer_pw2rs(rs, pw)
      TYPE(realspace_grid_type), INTENT(IN) :: rs
      TYPE(pw_r3d_rs_type), INTENT(IN) :: pw
      transfers = transfers + 1
      rs%buffer%host_buffer = RESHAPE(pw%array, [12])
   END SUBROUTINE
   SUBROUTINE transfer_rg(a, b)
      TYPE(pw_r3d_rs_type), INTENT(IN) :: a
      TYPE(pw_c1d_gs_type), INTENT(INOUT) :: b
      b%array = CMPLX(RESHAPE(a%array, [12]), 0.0_dp, dp)
   END SUBROUTINE
   SUBROUTINE transfer_gr(a, b)
      TYPE(pw_c1d_gs_type), INTENT(IN) :: a
      TYPE(pw_r3d_rs_type), INTENT(INOUT) :: b
      b%array = RESHAPE(REAL(a%array, dp), [3, 2, 2])
   END SUBROUTINE
   SUBROUTINE pw_axpy(a, b)
      TYPE(pw_c1d_gs_type), INTENT(IN) :: a
      TYPE(pw_c1d_gs_type), INTENT(INOUT) :: b
      b%array = b%array + a%array
   END SUBROUTINE
   SUBROUTINE copy_r(a, b)
      TYPE(pw_r3d_rs_type), INTENT(IN) :: a
      TYPE(pw_r3d_rs_type), INTENT(INOUT) :: b
      b%array = a%array
   END SUBROUTINE
   SUBROUTINE copy_g(a, b)
      TYPE(pw_c1d_gs_type), INTENT(IN) :: a
      TYPE(pw_c1d_gs_type), INTENT(INOUT) :: b
      b%array = a%array
   END SUBROUTINE
   SUBROUTINE zero_r(a)
      TYPE(pw_r3d_rs_type), INTENT(INOUT) :: a
      a%array = 0
   END SUBROUTINE
   SUBROUTINE zero_g(a)
      TYPE(pw_c1d_gs_type), INTENT(INOUT) :: a
      a%array = 0
   END SUBROUTINE
   SUBROUTINE pw_prolongate_s3(a, b, pool, section)
      TYPE(pw_r3d_rs_type) :: a, b
      TYPE(pool_type) :: pool
      INTEGER :: section
      ERROR STOP 'unexpected spline'
   END SUBROUTINE
   SUBROUTINE pw_restrict_s3(a, b, pool, section)
      TYPE(pw_r3d_rs_type) :: a, b
      TYPE(pool_type) :: pool
      INTEGER :: section
      ERROR STOP 'unexpected spline'
   END SUBROUTINE
   FUNCTION cp_to_string(i) RESULT(s)
      INTEGER :: i
      CHARACTER(10) :: s
      WRITE (s, '(I10)') i
   END FUNCTION
   SUBROUTINE cp_abort(location, message)
      CHARACTER(*) :: location, message
      ERROR STOP 'unexpected abort'
   END SUBROUTINE
   SUBROUTINE timeset(name, handle)
      CHARACTER(*) :: name
      INTEGER, INTENT(OUT) :: handle
      handle = 1
   END SUBROUTINE
   SUBROUTINE timestop(handle)
      INTEGER :: handle
   END SUBROUTINE
END MODULE

PROGRAM test_grid_bindings
   USE grid_fixture
   IMPLICIT NONE
   TYPE(pw_env_type) :: env
   TYPE(pw_grid_type), POINTER :: grid
   TYPE(realspace_grid_type) :: rs(1)
   TYPE(pw_r3d_rs_type) :: rho, potential
   TYPE(pw_c1d_gs_type) :: g
   REAL(dp) :: expected(12)
   INTEGER :: i, pass
   LOGICAL :: enabled
   ALLOCATE (grid)
   ALLOCATE (grid%gsq(12), grid%g_hatmap(12, 1))
   grid%gsq = 1
   grid%g_hatmap(:, 1) = [(i - 1, i=1, 12)]
   ALLOCATE (env%gridlevel_info, env%pools(1), rs(1)%desc)
   env%pools(1)%pool%grid => grid
   rs(1)%desc%pw => grid
   ALLOCATE (rho%array(3, 2, 2), potential%array(3, 2, 2), g%array(12))
   rho%pw_grid => grid
   potential%pw_grid => grid
   g%pw_grid => grid
   CALL grid_fixture_begin()
   CALL offload_create_buffer(12, rs(1)%buffer)
   enabled = pw_gpu_res_enabled(pw_gpu_res_density_id)
   expected = [(0.1_dp*i, i=1, 12)]
   DO pass = 1, 2
      rs(1)%buffer%host_buffer = expected
      IF (pass == 2) rs(1)%desc%border = 1 ! Host path must not borrow padded storage.
      CALL rs_pw_grid_prepare(rs)
      CALL grid_fixture_collocate(rs(1)%buffer%c_ptr)
      transfers = 0
      CALL density_rs2pw(env, rs, rho, g)
      IF (enabled .AND. pass == 1) THEN
         IF (transfers /= 0) ERROR STOP 'device density used host transfer'
      ELSE
         IF (transfers /= 1) ERROR STOP 'density fallback missing'
      END IF
      IF (MAXVAL(ABS(RESHAPE(rho%array, [12]) - expected)) > 1e-12_dp) ERROR STOP 'density value'
   END DO
   rs(1)%desc%border = 0
   potential%array = RESHAPE(expected, [3, 2, 2])
   CALL grid_fixture_potential(C_LOC(potential%array), 12_C_SIZE_T)
   transfers = 0
   rs(1)%buffer%host_buffer = -321
   CALL potential_pw2rs(rs, potential, env, device_grid=.TRUE.)
   IF (enabled) THEN
      IF (transfers /= 0) ERROR STOP 'borrowed potential used host transfer'
      IF (.NOT. C_ASSOCIATED(offload_grid_device(rs(1)%buffer, 12))) ERROR STOP 'missing device potential'
      IF (ANY(rs(1)%buffer%host_buffer /= -321)) ERROR STOP 'potential downloaded'
   END IF
   CALL offload_grid_host(rs(1)%buffer)
   IF (ANY(rs(1)%buffer%host_buffer /= expected)) ERROR STOP 'potential fallback value'
   transfers = 0
   CALL potential_pw2rs(rs, potential, env) ! General callers retain the host contract.
   IF (transfers /= 1) ERROR STOP 'general caller lost host output'
   IF (ANY(rs(1)%buffer%host_buffer /= expected)) ERROR STOP 'general potential value'
   CALL offload_free_buffer(rs(1)%buffer)
   CALL grid_fixture_end()
   DEALLOCATE (rho%array, potential%array, g%array, rs(1)%desc, env%pools, env%gridlevel_info)
   DEALLOCATE (grid%gsq, grid%g_hatmap)
   DEALLOCATE (grid)
   PRINT *, 'PASS Grid/PW production Fortran bindings and callers'
END PROGRAM
