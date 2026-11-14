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
   INTEGER, PARAMETER :: pw_gpu_res_density_id = 1, pw_gpu_res_hartree_id = 2, pw_gpu_res_pot_id = 4
   INTEGER, PARAMETER :: pw_poisson_multipole = 4, MULTIPOLE0D = 1303
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
      LOGICAL :: have_g0 = .TRUE.
      INTEGER :: first_gne0 = 2
      REAL(KIND=dp) :: dvol = 0.125_dp, vol = 1.0_dp
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
   TYPE section_vals_type
      REAL(dp) :: gcut = 2.0_dp
   END TYPE
   TYPE dft_control_type
      INTEGER :: nspins = 2
   END TYPE
   TYPE qs_rho_type
      TYPE(pw_c1d_gs_type), POINTER :: rho_g(:) => NULL()
   END TYPE
   TYPE qs_charges_type
      REAL(dp) :: total_rho_gspace = 0
   END TYPE
   TYPE qs_energy_type
      REAL(dp) :: hartree = 0
   END TYPE
   TYPE qs_environment_type
      TYPE(dft_control_type), POINTER :: dft => NULL()
      TYPE(qs_rho_type), POINTER :: rho => NULL()
      TYPE(qs_charges_type), POINTER :: charges => NULL()
      TYPE(pw_c1d_gs_type), POINTER :: core => NULL()
      TYPE(pw_r3d_rs_type), POINTER :: vh => NULL()
      TYPE(section_vals_type), POINTER :: input => NULL()
   END TYPE
   TYPE poisson_parameters_type
      INTEGER :: solver = pw_poisson_multipole
      LOGICAL :: has_dielectric = .FALSE.
   END TYPE
   TYPE green_type
      INTEGER :: method = MULTIPOLE0D
      TYPE(pw_c1d_gs_type) :: influence_fn
   END TYPE
   TYPE pool_type
      TYPE(pw_grid_type), POINTER :: pw_grid => NULL()
   END TYPE
   TYPE pool_pointer_type
      TYPE(pool_type), POINTER :: pool => NULL()
   END TYPE
   TYPE pw_poisson_type
      TYPE(poisson_parameters_type), POINTER :: parameters => NULL()
      TYPE(green_type), POINTER :: green_fft => NULL()
      TYPE(pool_pointer_type) :: pw_pools(1)
      INTEGER :: used_grid = 1, pw_level = 1
   END TYPE
   INTERFACE pw_gpu_res_ensure_host
      MODULE PROCEDURE pw_gpu_res_ensure_host_r, pw_gpu_res_ensure_host_g
   END INTERFACE
   INTERFACE pw_integrate_function
      MODULE PROCEDURE pw_integrate_function_r3d_rs, integrate_g
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
      SUBROUTINE res_fixture_spin_b(rho, g, n) BIND(C)
         IMPORT :: C_PTR, C_SIZE_T
         TYPE(C_PTR), VALUE :: rho, g
         INTEGER(C_SIZE_T), VALUE :: n
      END SUBROUTINE
      SUBROUTINE res_fixture_vxc_pair(va, vb, n) BIND(C)
         IMPORT :: C_PTR, C_SIZE_T
         TYPE(C_PTR), VALUE :: va, vb
         INTEGER(C_SIZE_T), VALUE :: n
      END SUBROUTINE
   END INTERFACE
CONTAINS
   #:include "res_bindings.inc"
   #:include "res_dispatch.inc"
   #:include "host_integrate.inc"
   #:include "host_copy.inc"
   #:include "host_axpy.inc"
   #:include "host_zero.inc"
   SUBROUTINE get_qs_env(env, dft_control, rho, rho_core, qs_charges, v_hartree_rspace, input)
      TYPE(qs_environment_type), POINTER :: env
      TYPE(dft_control_type), POINTER, OPTIONAL :: dft_control
      TYPE(qs_rho_type), POINTER, OPTIONAL :: rho
      TYPE(pw_c1d_gs_type), POINTER, OPTIONAL :: rho_core
      TYPE(qs_charges_type), POINTER, OPTIONAL :: qs_charges
      TYPE(pw_r3d_rs_type), POINTER, OPTIONAL :: v_hartree_rspace
      TYPE(section_vals_type), POINTER, OPTIONAL :: input
      IF (PRESENT(dft_control)) dft_control => env%dft
      IF (PRESENT(rho)) rho => env%rho
      IF (PRESENT(rho_core)) rho_core => env%core
      IF (PRESENT(qs_charges)) qs_charges => env%charges
      IF (PRESENT(v_hartree_rspace)) v_hartree_rspace => env%vh
      IF (PRESENT(input)) input => env%input
   END SUBROUTINE
   SUBROUTINE qs_rho_get(rho, rho_g)
      TYPE(qs_rho_type), POINTER :: rho
      TYPE(pw_c1d_gs_type), POINTER :: rho_g(:)
      rho_g => rho%rho_g
   END SUBROUTINE
   SUBROUTINE pw_poisson_rebuild(poisson, density)
      TYPE(pw_poisson_type), POINTER :: poisson
      TYPE(pw_c1d_gs_type), INTENT(IN) :: density
      IF (.NOT. ASSOCIATED(poisson%green_fft%influence_fn%pw_grid, density%pw_grid)) ERROR STOP 'Poisson grid'
   END SUBROUTINE
   REAL(dp) FUNCTION integrate_g(density, isign) RESULT(charge_value)
      TYPE(pw_c1d_gs_type), INTENT(IN) :: density
      INTEGER, INTENT(IN) :: isign
      charge_value = isign*REAL(density%array(1), dp)*density%pw_grid%vol
   END FUNCTION
   FUNCTION section_vals_get_subs_vals(section, name) RESULT(child)
      TYPE(section_vals_type), POINTER :: section, child
      CHARACTER(LEN=*), INTENT(IN) :: name
      IF (name /= "DFT%DENSITY_FITTING") ERROR STOP 'DDAPC section'
      child => section
   END FUNCTION
   SUBROUTINE section_vals_val_get(section, name, r_val)
      TYPE(section_vals_type), POINTER :: section
      CHARACTER(LEN=*), INTENT(IN) :: name
      REAL(dp), INTENT(OUT) :: r_val
      IF (name /= "GCUT") ERROR STOP 'DDAPC cutoff'
      r_val = section%gcut
   END SUBROUTINE
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
   TYPE(pw_r3d_rs_type), TARGET :: rho, copy, vh, rhob
   TYPE(pw_c1d_gs_type), TARGET :: g, gb, core, infl, gtot, spin_g(2)
   TYPE(qs_environment_type), POINTER :: qs_env
   TYPE(pw_poisson_type), POINTER :: poisson
   TYPE(qs_energy_type), POINTER :: energy
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
   ! Stage 3a: closing the SCF window writes a live density back.
   rho%array = expected
   CALL res_fixture_begin(C_LOC(rho%array), C_LOC(g%array), 12_C_SIZE_T)
   CALL pw_gpu_res_set_lazy(.TRUE.)
   CALL pw_gpu_res_defer_density(.TRUE.)
   CALL pw_gpu_res_defer_density(.FALSE.)
   CALL pw_gpu_res_discard(copy) ! not resident: no effect
   CALL pw_gpu_res_set_lazy(.FALSE.)
   IF (ANY(rho%array /= expected)) ERROR STOP 'window exit synchronization'
   CALL res_fixture_end()
   ! Releasing an array drops its device copy without a download.
   rho%array = expected
   CALL res_fixture_begin(C_LOC(rho%array), C_LOC(g%array), 12_C_SIZE_T)
   CALL pw_gpu_res_discard(rho)
   CALL pw_gpu_res_ensure_host(rho)
   IF (enabled .AND. ANY(rho%array /= 0)) ERROR STOP 'discard downloaded'
   IF (.NOT. enabled .AND. ANY(rho%array /= expected)) ERROR STOP 'discard changed host'
   CALL pw_gpu_res_ensure_host(g)
   CALL res_fixture_end()
   ALLOCATE (rhob%array(3, 2, 2), gb%array(12), core%array(12), infl%array(12), gtot%array(12))
   rhob%pw_grid => grid
   gb%pw_grid => grid
   core%pw_grid => grid
   infl%pw_grid => grid
   gtot%pw_grid => grid
   grid%g_hatmap(:, 1) = [(i - 1, i=1, 12)]
   rho%array = expected
   rhob%array = 2*expected
   g%array = (2.0_dp, 3.0_dp)
   gb%array = (5.0_dp, 7.0_dp)
   core%array = (1.0_dp, 0.0_dp)
   infl%array = (0.2_dp, 0.0_dp)
   gtot%array = (-99.0_dp, 0.0_dp)
   CALL res_fixture_begin(C_LOC(rho%array), C_LOC(g%array), 12_C_SIZE_T)
   CALL pw_gpu_res_allow_density()
   CALL res_fixture_spin_b(C_LOC(rhob%array), C_LOC(gb%array), 12_C_SIZE_T)
   CALL charge(rhob, total)
   IF (ABS(total + 2*SUM(expected)*grid%dvol) > 1e-12_dp) ERROR STOP 'UKS beta charge'
   IF (enabled .AND. pw_gpu_uks_enabled() .AND. ANY(rhob%array /= 0)) ERROR STOP 'UKS beta charge download'
   done = pw_gpu_res_hartree_uks_begin(g, gb, core, infl, 1, gtot)
   IF (done .NEQV. (enabled .AND. pw_gpu_uks_enabled())) ERROR STOP 'UKS Hartree binding'
   IF (done) THEN
      IF (gtot%array(1) /= CMPLX(8.0_dp, 10.0_dp, dp)) ERROR STOP 'UKS Hartree prefix'
      IF (ANY(gtot%array(2:) /= CMPLX(-99.0_dp, 0.0_dp, dp))) ERROR STOP 'UKS prefix overrun'
      IF (.NOT. pw_gpu_res_hartree_energy(total)) ERROR STOP 'UKS Hartree energy binding'
      IF (ABS(total - 393.6_dp) > 1e-12_dp) ERROR STOP 'UKS Hartree energy'
      IF (.NOT. pw_gpu_res_hartree_finish(vh, grid)) ERROR STOP 'UKS Hartree finish binding'
   END IF
   CALL pw_gpu_res_ensure_host_g(gb)
   CALL pw_gpu_res_ensure_host(rhob)
   IF (ANY(gb%array /= CMPLX(5.0_dp, 7.0_dp, dp))) ERROR STOP 'UKS G host synchronization'
   IF (ANY(rhob%array /= 2*expected)) ERROR STOP 'UKS real host synchronization'
   CALL pw_gpu_res_ensure_host(rho)
   CALL pw_gpu_res_ensure_host(g)
   CALL pw_gpu_res_ensure_host(vh)
   ALLOCATE (qs_env, poisson, energy)
   ALLOCATE (qs_env%dft, qs_env%rho, qs_env%charges, qs_env%input)
   ALLOCATE (poisson%parameters, poisson%green_fft, poisson%pw_pools(1)%pool)
   poisson%green_fft%influence_fn = infl
   poisson%pw_pools(1)%pool%pw_grid => grid
   spin_g(1) = g
   spin_g(2) = gb
   qs_env%rho%rho_g => spin_g
   qs_env%core => core
   qs_env%vh => vh
   CALL qs_ks_periodic_device(qs_env, poisson, gtot, energy, done)
   IF (done .NEQV. (enabled .AND. pw_gpu_uks_enabled())) ERROR STOP 'UKS periodic orchestration'
   IF (done) THEN
      IF (ABS(energy%hartree - 196.8_dp) > 1e-12_dp) ERROR STOP 'UKS periodic energy'
      IF (qs_env%charges%total_rho_gspace /= -8.0_dp) ERROR STOP 'UKS periodic charge'
   END IF
   CALL qs_ks_ddapc_device_begin(qs_env, poisson, gtot, done)
   IF (done .NEQV. (enabled .AND. pw_gpu_uks_enabled())) ERROR STOP 'UKS DDAPC orchestration'
   CALL pw_gpu_res_hartree_cancel()
   poisson%parameters%has_dielectric = .TRUE.
   CALL qs_ks_periodic_device(qs_env, poisson, gtot, energy, done)
   IF (done) ERROR STOP 'dielectric device path'
   poisson%parameters%has_dielectric = .FALSE.
   poisson%used_grid = 0
   CALL qs_ks_periodic_device(qs_env, poisson, gtot, energy, done)
   IF (done) ERROR STOP 'non-G-space device path'
   poisson%used_grid = 1
   qs_env%dft%nspins = 1
   qs_env%rho%rho_g => spin_g(1:1)
   CALL qs_ks_periodic_device(qs_env, poisson, gtot, energy, done)
   IF (done .NEQV. enabled) ERROR STOP 'RKS periodic orchestration'
   IF (done) THEN
      IF (ABS(energy%hartree - 21.6_dp) > 1e-12_dp) ERROR STOP 'RKS periodic energy'
      IF (qs_env%charges%total_rho_gspace /= -3.0_dp) ERROR STOP 'RKS periodic charge'
   END IF
   CALL pw_gpu_res_ensure_host(vh)
   DEALLOCATE (poisson%green_fft, poisson%parameters, poisson%pw_pools(1)%pool)
   DEALLOCATE (qs_env%dft, qs_env%rho, qs_env%charges, qs_env%input)
   DEALLOCATE (poisson, energy, qs_env)
   rho%array = expected
   rhob%array = 2*expected
   vh%array = 7
   CALL res_fixture_vxc_pair(C_LOC(rho%array), C_LOC(rhob%array), 12_C_SIZE_T)
   IF (enabled .AND. pw_gpu_uks_enabled()) THEN
      IF (.NOT. pw_gpu_res_vtot_prepare(vh, rho, grid%dvol)) ERROR STOP 'UKS Vtot alpha prepare'
      IF (.NOT. pw_gpu_res_vtot_skip_host()) ERROR STOP 'UKS Vtot alpha skip'
      IF (.NOT. pw_gpu_res_vtot_prepare(vh, rhob, grid%dvol)) ERROR STOP 'UKS Vtot beta prepare'
      IF (.NOT. pw_gpu_res_vtot_skip_host()) ERROR STOP 'UKS Vtot beta skip'
      CALL pw_gpu_res_ensure_host(rho)
      CALL pw_gpu_res_ensure_host(rhob)
      IF (ANY(rho%array /= 7 + grid%dvol*expected)) ERROR STOP 'UKS Vtot alpha'
      IF (ANY(rhob%array /= 7 + 2*grid%dvol*expected)) ERROR STOP 'UKS Vtot beta'
   END IF
   CALL res_fixture_end()
   DEALLOCATE (rhob%array, gb%array, core%array, infl%array, gtot%array)
   DEALLOCATE (grid%gsq, grid%g_hatmap)
   DEALLOCATE (grid, rho%array, copy%array, vh%array, g%array)
   PRINT *, 'PASS residency Fortran bindings'
END PROGRAM
