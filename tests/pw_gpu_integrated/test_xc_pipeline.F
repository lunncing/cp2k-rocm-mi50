! SPDX-License-Identifier: GPL-2.0-or-later
#define CPASSERT(condition) IF (.NOT. (condition)) ERROR STOP 'CPASSERT'
#define CPABORT(message) ERROR STOP message
MODULE xc_pipeline_fixture
   USE ISO_C_BINDING
   USE, INTRINSIC :: ieee_arithmetic
   IMPLICIT NONE
   INTEGER, PARAMETER :: dp = C_DOUBLE, PW_MODE_LOCAL = 1
   INTEGER, PARAMETER :: xc_deriv_pw = 1, xc_rho_no_smooth = 1, xc_pbe_orig = 1, xc_pbe_rev = 2, xc_pbe_sol = 3
   INTEGER, PARAMETER :: deriv_norm_drhoa = 1, deriv_norm_drhob = 2
   REAL(KIND=dp), PARAMETER :: pi = 3.141592653589793238462643383279502884_dp
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
   LOGICAL, PARAMETER :: gpu_enabled = .TRUE.
#else
   LOGICAL, PARAMETER :: gpu_enabled = .FALSE.
#endif
   INTEGER :: cpu_calls = 0, timers = 0
   TYPE para_type
      INTEGER :: mode = PW_MODE_LOCAL
   END TYPE
   TYPE pw_grid_type
      TYPE(para_type) :: para
      INTEGER :: id_nr = 1, bounds_local(2, 3) = RESHAPE([-1, 1, 2, 4, 0, 2], [2, 3])
      INTEGER :: npts(3) = [3, 3, 3]
      INTEGER, POINTER :: g_hatmap(:, :) => NULL()
      REAL(KIND=dp), POINTER :: g(:, :) => NULL(), gsq(:) => NULL()
      REAL(KIND=dp) :: dvol = .37_dp
      LOGICAL :: spherical = .FALSE.
   END TYPE
   TYPE pw_r3d_rs_type
      TYPE(pw_grid_type), POINTER :: pw_grid => NULL()
      REAL(KIND=dp), POINTER :: array(:, :, :) => NULL()
   END TYPE
   TYPE pw_c1d_gs_type
      TYPE(pw_grid_type), POINTER :: pw_grid => NULL()
      COMPLEX(KIND=dp), POINTER :: array(:) => NULL()
   END TYPE
   TYPE pw_pool_type
      TYPE(pw_grid_type), POINTER :: grid => NULL()
      INTEGER :: outstanding = 0
   CONTAINS
      PROCEDURE :: create_pw, give_back_pw
   END TYPE
   TYPE section_name_type
      CHARACTER(LEN=12) :: name = 'PBE'
   END TYPE
   TYPE section_vals_type
      TYPE(section_name_type) :: section
      TYPE(section_vals_type), POINTER :: children(:) => NULL()
      INTEGER :: deriv = xc_deriv_pw, smooth = xc_rho_no_smooth, param = xc_pbe_orig
      LOGICAL :: active = .TRUE.
      REAL(KIND=dp) :: sc = 1, sx = 1, rho_cutoff = 1e-10_dp, drho_cutoff = .02_dp, smooth_range = 0
   END TYPE
   ! Unused CPU-only locals in the extracted early-dispatch portion of xc.F.
   TYPE cp_sll_xc_deriv_type
      INTEGER :: unused
   END TYPE
   TYPE xc_derivative_set_type
      INTEGER :: unused
   END TYPE
   TYPE xc_derivative_type
      INTEGER :: unused
   END TYPE
   TYPE xc_rho_set_type
      INTEGER :: unused
   END TYPE
   INTERFACE
      SUBROUTINE test_gpu_begin(limited) BIND(C)
         IMPORT C_INT
         INTEGER(C_INT), VALUE :: limited
      END SUBROUTINE
      SUBROUTINE test_gpu_end() BIND(C)
      END SUBROUTINE
      FUNCTION test_kernel_calls(kind) BIND(C) RESULT(n)
         IMPORT C_INT
         INTEGER(C_INT), VALUE :: kind
         INTEGER(C_INT) :: n
      END FUNCTION
   END INTERFACE
CONTAINS
#include "xc_bindings.inc"
#include "xc_dispatch.inc"
#include "xc_cutoff.inc"
#include "xc_cpu_pbe.inc"
   SUBROUTINE timeset(name, handle)
      CHARACTER(LEN=*), INTENT(IN) :: name
      INTEGER, INTENT(OUT) :: handle
      handle = LEN_TRIM(name)
      timers = timers + 1
   END SUBROUTINE
   SUBROUTINE timestop(handle)
      INTEGER, INTENT(IN) :: handle
      IF (handle <= 0) ERROR STOP 'timer'
      timers = timers - 1
   END SUBROUTINE
   SUBROUTINE create_pw(pool, pw)
      CLASS(pw_pool_type) :: pool
      TYPE(pw_r3d_rs_type) :: pw
      pw%pw_grid => pool%grid
      ALLOCATE (pw%array(-1:1, 2:4, 0:2))
      pw%array = -987.0_dp
      pool%outstanding = pool%outstanding + 1
   END SUBROUTINE
   SUBROUTINE give_back_pw(pool, pw)
      CLASS(pw_pool_type) :: pool
      TYPE(pw_r3d_rs_type) :: pw
      DEALLOCATE (pw%array)
      NULLIFY (pw%pw_grid)
      pool%outstanding = pool%outstanding - 1
   END SUBROUTINE
   INTEGER FUNCTION section_get_ival(section, name) RESULT(value)
      TYPE(section_vals_type), POINTER :: section
      CHARACTER(LEN=*), INTENT(IN) :: name
      SELECT CASE (name)
      CASE ('XC_GRID%XC_DERIV')
         value = section%deriv
      CASE ('XC_GRID%XC_SMOOTH_RHO')
         value = section%smooth
      CASE DEFAULT
         ERROR STOP 'unknown integer keyword'
      END SELECT
   END FUNCTION
   REAL(KIND=dp) FUNCTION section_get_rval(section, name) RESULT(value)
      TYPE(section_vals_type), POINTER :: section
      CHARACTER(LEN=*), INTENT(IN) :: name
      SELECT CASE (name)
      CASE ('DENSITY_CUTOFF')
         value = section%rho_cutoff
      CASE ('GRADIENT_CUTOFF')
         value = section%drho_cutoff
      CASE ('DENSITY_SMOOTH_CUTOFF_RANGE')
         value = section%smooth_range
      CASE DEFAULT
         ERROR STOP 'unknown real keyword'
      END SELECT
   END FUNCTION
   SUBROUTINE section_vals_val_get(section, name, i_val, r_val, l_val)
      TYPE(section_vals_type), POINTER :: section
      CHARACTER(LEN=*), INTENT(IN) :: name
      INTEGER, OPTIONAL, INTENT(OUT) :: i_val
      REAL(KIND=dp), OPTIONAL, INTENT(OUT) :: r_val
      LOGICAL, OPTIONAL, INTENT(OUT) :: l_val
      SELECT CASE (name)
      CASE ('PARAMETRIZATION')
         i_val = section%param
      CASE ('SCALE_C')
         r_val = section%sc
      CASE ('SCALE_X')
         r_val = section%sx
      CASE ('_SECTION_PARAMETERS_')
         l_val = section%active
      CASE DEFAULT
         ERROR STOP 'unknown section keyword'
      END SELECT
   END SUBROUTINE
   FUNCTION section_vals_get_subs_vals(section, name) RESULT(child)
      TYPE(section_vals_type), POINTER :: section, child
      CHARACTER(LEN=*), INTENT(IN) :: name
      IF (name /= 'XC_FUNCTIONAL') ERROR STOP 'unknown subsection'
      child => section%children(1)
   END FUNCTION
   FUNCTION section_vals_get_subs_vals2(section, i_section) RESULT(child)
      TYPE(section_vals_type), POINTER :: section, child
      INTEGER, INTENT(IN) :: i_section
      NULLIFY (child)
      IF (i_section <= SIZE(section%children)) child => section%children(i_section)
   END FUNCTION

   ! Independent direct DFT in Fortran, with CP2K's x-fastest real-grid order.
   FUNCTION dft(input, direction) RESULT(output)
      COMPLEX(KIND=dp), INTENT(IN) :: input(27)
      INTEGER, INTENT(IN) :: direction
      COMPLEX(KIND=dp) :: output(27)
      INTEGER :: j, k, jv(3), kv(3)
      REAL(KIND=dp) :: angle
      output = 0
      DO k = 0, 26
         kv = [MOD(k, 3), MOD(k/3, 3), k/9]
         DO j = 0, 26
            jv = [MOD(j, 3), MOD(j/3, 3), j/9]
            angle = direction*2*pi*SUM(jv*kv)/3
            output(k + 1) = output(k + 1) + input(j + 1)*CMPLX(COS(angle), SIN(angle), dp)
         END DO
      END DO
   END FUNCTION
   FUNCTION scatter(z, grid) RESULT(full)
      COMPLEX(KIND=dp), INTENT(IN) :: z(8)
      TYPE(pw_grid_type), INTENT(IN) :: grid
      COMPLEX(KIND=dp) :: full(27)
      INTEGER :: i
      full = 0
      DO i = 1, 8
         full(grid%g_hatmap(i, 1) + 1) = z(i)
         IF (SIZE(grid%g_hatmap, 2) == 2) full(grid%g_hatmap(i, 2) + 1) = CONJG(z(i))
      END DO
   END FUNCTION
   SUBROUTINE reference(rho_r, rho_g, section, result, energy)
      TYPE(pw_r3d_rs_type), INTENT(IN) :: rho_r
      TYPE(pw_c1d_gs_type), INTENT(IN) :: rho_g
      TYPE(section_vals_type), POINTER :: section
      REAL(KIND=dp), INTENT(OUT) :: result(27), energy
      REAL(KIND=dp), TARGET :: rho(27), norm(27), e(27, 10), grad(27, 3)
      REAL(KIND=dp), POINTER :: ep(:, :, :), rp(:, :, :), null_array(:, :, :)
      COMPLEX(KIND=dp) :: full(27), z(8), divg(8)
      TYPE(pw_grid_type), POINTER :: grid
      TYPE(section_vals_type), POINTER :: pbe
      INTEGER :: d
      grid => rho_r%pw_grid
      pbe => section%children(1)%children(1)
      rho = RESHAPE(rho_r%array, [27])
      DO d = 1, 3
         z = CMPLX(0.0_dp, 1.0_dp, dp)*grid%g(d, :)*rho_g%array
         full = dft(scatter(z, grid), 1)
         grad(:, d) = REAL(full, dp)
      END DO
      norm = SQRT(grad(:, 1)**2 + grad(:, 2)**2 + grad(:, 3)**2)
      e = 0
!$OMP PARALLEL
      CALL cpu_pbe_lda_calc(rho, norm, e(:, 1), e(:, 2), e(:, 3), e(:, 4), e(:, 5), e(:, 6), &
                            e(:, 7), e(:, 8), e(:, 9), e(:, 10), 1, 27, section%rho_cutoff, &
                            pbe%param, pbe%sc, pbe%sx)
!$OMP END PARALLEL
      divg = 0
      DO d = 1, 3
         grad(:, d) = -(grad(:, d)*(e(:, 3)/MAX(norm, section%drho_cutoff)))
         full = dft(CMPLX(grad(:, d), 0.0_dp, dp), -1)
         z = full(grid%g_hatmap(:, 1) + 1)/27
         divg = divg + CMPLX(0.0_dp, 1.0_dp, dp)*grid%g(d, :)*z
      END DO
      full = dft(scatter(divg, grid), 1)
      e(:, 2) = e(:, 2) + REAL(full, dp)
      IF (grid%spherical) THEN
         full = dft(CMPLX(e(:, 2), 0.0_dp, dp), -1)
         z = full(grid%g_hatmap(:, 1) + 1)/27
         full = dft(scatter(z, grid), 1)
         e(:, 2) = REAL(full, dp)
      END IF
      NULLIFY (null_array)
      rp(1:3, 1:3, 1:3) => rho
      ep(1:3, 1:3, 1:3) => e(:, 1)
      CALL smooth_cutoff(ep, rp, null_array, null_array, section%rho_cutoff, section%smooth_range)
      energy = SUM(e(:, 1))*grid%dvol
      ep(1:3, 1:3, 1:3) => e(:, 2)
      CALL smooth_cutoff(ep, rp, null_array, null_array, &
                         section%rho_cutoff*section%smooth_range, section%smooth_range)
      result = e(:, 2)
   END SUBROUTINE
END MODULE

PROGRAM test_xc_pipeline
   USE xc_pipeline_fixture
   IMPLICIT NONE
   TYPE(pw_grid_type), TARGET :: grid, other_grid
   TYPE(pw_pool_type), POINTER :: pool
   TYPE(section_vals_type), POINTER :: section, pbe, other
   TYPE(pw_r3d_rs_type), POINTER :: rho(:), tau(:), vxc(:), vtau(:), weights
   TYPE(pw_c1d_gs_type), POINTER :: rho_g(:), saved_g(:)
   TYPE(pw_r3d_rs_type), TARGET :: energy_grid
   INTEGER :: i, d, j, nmaps, filter, param, variant, before, calls, skip, x, y, z
   INTEGER, PARAMETER :: modes(8) = [0, 1, 3, 4, 9, 10, 12, 13]
   REAL(KIND=dp) :: exc, expected_energy, virial(3, 3), expected(27), density(27)
   REAL(KIND=dp), POINTER :: saved_density(:, :, :)
   ALLOCATE (pool, section, rho(1), rho_g(1))
   ALLOCATE (section%children(1), grid%g(3, 8), grid%gsq(8))
   ALLOCATE (section%children(1)%children(2))
   pbe => section%children(1)%children(1)
   other => section%children(1)%children(2)
   other%section%name = 'BLYP'
   other%active = .FALSE.
   pool%grid => grid
   NULLIFY (tau, vxc, vtau, weights)
   rho(1)%pw_grid => grid
   rho_g(1)%pw_grid => grid
   ALLOCATE (rho(1)%array(-1:1, 2:4, 0:2), rho_g(1)%array(8))
   DO i = 1, 8
      x = MOD(modes(i), 3)
      y = MOD(modes(i)/3, 3)
      z = modes(i)/9
      grid%g(:, i) = [.2_dp*x, .3_dp*y, .4_dp*z]
      rho_g(1)%array(i) = CMPLX(.03_dp*COS(REAL(i - 1, dp)), .02_dp*SIN(REAL(i - 1, dp)), dp)
   END DO
   CALL test_gpu_begin(0_C_INT)
   DO nmaps = 1, 2
      ALLOCATE (grid%g_hatmap(8, nmaps))
      grid%g_hatmap(:, 1) = modes
      IF (nmaps == 2) THEN
         DO i = 1, 8
            grid%g_hatmap(i, 2) = MOD(3 - MOD(modes(i), 3), 3) + &
                                  3*MOD(3 - MOD(modes(i)/3, 3), 3) + 9*MOD(3 - modes(i)/9, 3)
         END DO
      END IF
      DO filter = 0, 1
         grid%spherical = filter == 1
         DO param = 1, 3
            pbe%param = param
            DO variant = 0, 3
               pbe%sc = .7_dp
               pbe%sx = .9_dp
               section%rho_cutoff = 1e-10_dp
               section%smooth_range = 0
               density = [(.7_dp + .2_dp*SIN(.17_dp*i), i=1, 27)]
               DO i = 1, 8
                  rho_g(1)%array(i) = CMPLX(.03_dp*COS(REAL(i - 1, dp)), .02_dp*SIN(REAL(i - 1, dp)), dp)
               END DO
               IF (variant == 1) THEN
                  section%rho_cutoff = .3_dp
                  section%smooth_range = 2
               ELSEIF (variant == 2) THEN
                  pbe%sc = 0
                  density(1:5) = [-1.0_dp, 0.0_dp, 1e-10_dp, NEAREST(1e-10_dp, -1.0_dp), &
                                   NEAREST(1e-10_dp, 1.0_dp)]
                  density(6) = ieee_value(0.0_dp, ieee_quiet_nan)
               ELSEIF (variant == 3) THEN
                  pbe%sx = 0
                  density = .7_dp
                  rho_g(1)%array = 0
               END IF
               rho(1)%array = RESHAPE(density, [3, 3, 3])
               CALL reference(rho(1), rho_g(1), section, expected, expected_energy)
               IF (.NOT. ALL(ieee_is_finite(expected)) .OR. .NOT. ieee_is_finite(expected_energy)) &
                  ERROR STOP 'nonfinite XC reference'
               before = test_kernel_calls(2_C_INT)
               calls = cpu_calls
               CALL xc_vxc_pw_create(vxc, vtau, exc, rho, rho_g, tau, section, weights, pool, .FALSE., virial)
               IF (gpu_enabled) THEN
                  IF (cpu_calls /= calls .OR. .NOT. ASSOCIATED(vxc)) ERROR STOP 'fast path not taken'
                  IF (test_kernel_calls(2_C_INT) /= before + 1) ERROR STOP 'PBE launch count'
                  IF (.NOT. ALL(ieee_is_finite(vxc(1)%array)) .OR. .NOT. ieee_is_finite(exc)) &
                     ERROR STOP 'nonfinite full XC result'
                  IF (ANY(ABS(RESHAPE(vxc(1)%array, [27]) - expected) > 1e-12_dp*(1 + ABS(expected)))) &
                     ERROR STOP 'full XC potential differs from Fortran reference'
                  IF (ABS(exc - expected_energy) > 1e-12_dp*(1 + ABS(expected_energy))) &
                     ERROR STOP 'full XC energy differs from Fortran reference'
                  IF (ASSOCIATED(vtau) .OR. ANY(virial /= 0)) ERROR STOP 'unused outputs'
                  CALL pool%give_back_pw(vxc(1))
                  DEALLOCATE (vxc)
               ELSE
                  IF (cpu_calls /= calls + 1 .OR. ASSOCIATED(vxc)) ERROR STOP 'CPU dispatch'
               END IF
               IF (pool%outstanding /= 0 .OR. timers /= 0) ERROR STOP 'pool/timer leak'
            END DO
         END DO
      END DO
      CALL test_gpu_end()
      DEALLOCATE (grid%g_hatmap)
      IF (nmaps == 1) CALL test_gpu_begin(0_C_INT)
   END DO
   ALLOCATE (grid%g_hatmap(8, 1))
   grid%g_hatmap(:, 1) = modes
   CALL test_gpu_begin(0_C_INT)
   ! Exercise the actual pre-derivative dispatch guards and failed-attempt cleanup.
   DO skip = 1, 13
      section%deriv = xc_deriv_pw
      section%smooth = xc_rho_no_smooth
      pbe%active = .TRUE.
      pbe%param = xc_pbe_orig
      other%active = .FALSE.
      grid%para%mode = PW_MODE_LOCAL
      SELECT CASE (skip)
      CASE (1)
         section%deriv = 2
      CASE (2)
         section%smooth = 2
      CASE (3)
         other%active = .TRUE.
      CASE (4)
         pbe%active = .FALSE.
      CASE (5)
         pbe%param = 99
      CASE (6)
         grid%para%mode = 2
      CASE (7)
         weights => energy_grid
      CASE (8)
         rho_g(1)%pw_grid => other_grid
      CASE (9)
         saved_g => rho_g
         NULLIFY (rho_g)
      CASE (12)
         saved_density => rho(1)%array
         ALLOCATE (rho(1)%array(1, 9, 3))
         rho(1)%array = .7_dp
      CASE (13)
         CALL test_gpu_end()
         CALL test_gpu_begin(1_C_INT)
      END SELECT
      calls = cpu_calls
      before = test_kernel_calls(2_C_INT)
      IF (skip == 10) THEN
         CALL xc_vxc_pw_create(vxc, vtau, exc, rho, rho_g, tau, section, weights, pool, .FALSE., virial, energy_grid)
      ELSE
         CALL xc_vxc_pw_create(vxc, vtau, exc, rho, rho_g, tau, section, weights, pool, skip == 11, virial)
      END IF
      IF (cpu_calls /= calls + 1 .OR. test_kernel_calls(2_C_INT) /= before) ERROR STOP 'unsupported dispatch'
      IF (ASSOCIATED(vxc) .OR. pool%outstanding /= 0 .OR. timers /= 0) ERROR STOP 'fallback cleanup'
      NULLIFY (weights)
      IF (skip == 9) rho_g => saved_g
      IF (skip == 12) THEN
         DEALLOCATE (rho(1)%array)
         rho(1)%array => saved_density
      END IF
      rho_g(1)%pw_grid => grid
   END DO
   CALL test_gpu_end()
   DEALLOCATE (grid%g_hatmap, grid%g, grid%gsq, rho(1)%array, rho_g(1)%array)
   DEALLOCATE (rho, rho_g, section%children(1)%children)
   DEALLOCATE (section%children)
   DEALLOCATE (section, pool)
   PRINT *, 'PASS full XC Fortran reference and dispatch'
END PROGRAM
