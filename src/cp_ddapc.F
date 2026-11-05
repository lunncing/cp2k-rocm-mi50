!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Density Derived atomic point charges from a QM calculation
!>      (see Bloechl, J. Chem. Phys. Vol. 103 pp. 7422-7428)
!> \par History
!>      08.2005 created [tlaino]
!> \author Teodoro Laino
! **************************************************************************************************
MODULE cp_ddapc
   USE bibliography,                    ONLY: Blochl1995,&
                                              cite_reference
   USE cell_types,                      ONLY: cell_type
   USE cp_control_types,                ONLY: ddapc_restraint_type,&
                                              dft_control_type
   USE cp_dbcsr_api,                    ONLY: dbcsr_copy,&
                                              dbcsr_p_type,&
                                              dbcsr_set
   USE cp_ddapc_forces,                 ONLY: ewald_ddapc_force,&
                                              reset_ch_pulay,&
                                              restraint_functional_force,&
                                              solvation_ddapc_force
   USE cp_ddapc_util,                   ONLY: get_ddapc,&
                                              modify_hartree_pot,&
                                              restraint_functional_potential
   USE cp_log_handling,                 ONLY: cp_get_default_logger,&
                                              cp_logger_type
   USE cp_output_handling,              ONLY: cp_print_key_finished_output,&
                                              cp_print_key_unit_nr
   USE input_constants,                 ONLY: do_spin_density
   USE input_section_types,             ONLY: section_vals_get_subs_vals,&
                                              section_vals_type,&
                                              section_vals_val_get
   USE kinds,                           ONLY: dp
   USE particle_types,                  ONLY: particle_type
   USE pw_gpu,                          ONLY: pw_gpu_res_enabled,&
                                              pw_gpu_res_hartree_begin,&
                                              pw_gpu_res_hartree_cancel,&
                                              pw_gpu_res_hartree_energy,&
                                              pw_gpu_res_hartree_finish,&
                                              pw_gpu_res_hartree_id
   USE pw_grid_types,                   ONLY: PW_MODE_LOCAL,&
                                              pw_grid_type
   USE pw_methods,                      ONLY: pw_integral_ab,&
                                              pw_integrate_function,&
                                              pw_scale,&
                                              pw_transfer,&
                                              pw_zero
   USE pw_poisson_methods,              ONLY: pw_poisson_rebuild
   USE pw_poisson_types,                ONLY: MULTIPOLE0D,&
                                              pw_poisson_multipole,&
                                              pw_poisson_type
   USE pw_pool_types,                   ONLY: pw_pool_type
   USE pw_types,                        ONLY: pw_c1d_gs_type,&
                                              pw_r3d_rs_type
   USE qs_energy_types,                 ONLY: qs_energy_type
   USE qs_charges_types,                ONLY: qs_charges_type
   USE qs_environment_types,            ONLY: get_qs_env,&
                                              qs_environment_type
   USE qs_integrate_potential,          ONLY: integrate_v_rspace
   USE qs_rho_types,                    ONLY: qs_rho_get,&
                                              qs_rho_type
#include "./base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   LOGICAL, PRIVATE, PARAMETER :: debug_this_module = .FALSE.
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'cp_ddapc'

   PUBLIC :: cp_ddapc_apply_CD, & ! Apply Coupling/Decoupling to Periodic Images
             qs_ks_ddapc, &
             qs_ks_ddapc_device_begin, qs_ks_ddapc_device_finish

CONTAINS

! **************************************************************************************************
!> \brief Set of methods using DDAPC charges
!> \param qs_env ...
!> \param auxbas_pw_pool ...
!> \param rho_tot_gspace ...
!> \param v_hartree_gspace ...
!> \param v_spin_ddapc_rest_r ...
!> \param energy ...
!> \param calculate_forces ...
!> \param ks_matrix ...
!> \param just_energy ...
!> \par History
!>      08.2005 created [tlaino]
!>      08.2008 extended to restraint/constraint DDAPC charges [fschiff]
! **************************************************************************************************
   SUBROUTINE qs_ks_ddapc(qs_env, auxbas_pw_pool, rho_tot_gspace, v_hartree_gspace, &
                          v_spin_ddapc_rest_r, energy, calculate_forces, ks_matrix, just_energy)

      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(pw_pool_type), POINTER                        :: auxbas_pw_pool
      TYPE(pw_c1d_gs_type), INTENT(IN)                   :: rho_tot_gspace, v_hartree_gspace
      TYPE(pw_r3d_rs_type), POINTER                      :: v_spin_ddapc_rest_r
      TYPE(qs_energy_type), POINTER                      :: energy
      LOGICAL, INTENT(in)                                :: calculate_forces
      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: ks_matrix
      LOGICAL, INTENT(in)                                :: just_energy

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'qs_ks_ddapc'

      INTEGER                                            :: ddapc_size, handle, i, my_id
      LOGICAL                                            :: ddapc_restraint_is_spin, &
                                                            et_coupling_calc, explicit_potential
      TYPE(cp_logger_type), POINTER                      :: logger
      TYPE(ddapc_restraint_type), POINTER                :: ddapc_restraint_control
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(pw_c1d_gs_type)                               :: v_spin_ddapc_rest_g
      TYPE(pw_r3d_rs_type), POINTER                      :: v_hartree_rspace

      NULLIFY (v_hartree_rspace, dft_control)

      CALL timeset(routineN, handle)
      CALL cite_reference(Blochl1995)
      ! In case decouple periodic images and/or apply restraints to charges
      logger => cp_get_default_logger()
      ddapc_restraint_is_spin = .FALSE.
      et_coupling_calc = .FALSE.
      ddapc_size = 0

      ! no k-points
      CPASSERT(SIZE(ks_matrix, 2) == 1)

      CALL get_qs_env(qs_env, &
                      v_hartree_rspace=v_hartree_rspace, &
                      dft_control=dft_control)

      IF (dft_control%qs_control%ddapc_restraint) THEN
         ddapc_size = SIZE(dft_control%qs_control%ddapc_restraint_control)
         IF (SIZE(energy%ddapc_restraint) /= ddapc_size) THEN
            DEALLOCATE (energy%ddapc_restraint)
            ALLOCATE (energy%ddapc_restraint(ddapc_size))
         END IF

         DO i = 1, SIZE(dft_control%qs_control%ddapc_restraint_control)
            my_id = dft_control%qs_control%ddapc_restraint_control(i)%density_type
            IF (my_id == do_spin_density .OR. ddapc_restraint_is_spin) ddapc_restraint_is_spin = .TRUE.
         END DO
         et_coupling_calc = dft_control%qs_control%et_coupling_calc
      END IF

      explicit_potential = ddapc_restraint_is_spin .OR. et_coupling_calc
      dft_control%qs_control%ddapc_explicit_potential = explicit_potential
      dft_control%qs_control%ddapc_restraint_is_spin = ddapc_restraint_is_spin
      IF (explicit_potential) THEN
         CALL auxbas_pw_pool%create_pw(v_spin_ddapc_rest_g)
         CALL pw_zero(v_spin_ddapc_rest_g)
         NULLIFY (v_spin_ddapc_rest_r)
         ALLOCATE (v_spin_ddapc_rest_r)
         CALL auxbas_pw_pool%create_pw(v_spin_ddapc_rest_r)
      END IF

      IF (calculate_forces) CALL reset_ch_pulay(qs_env)

      ! Decoupling/Recoupling
      CALL cp_ddapc_apply_CD(qs_env, rho_tot_gspace, energy%hartree, v_hartree_gspace, &
                             calculate_forces, Itype_of_density="FULL DENSITY")
      IF (dft_control%qs_control%ddapc_restraint) THEN
         ! Restraints/Constraints
         DO i = 1, ddapc_size
            NULLIFY (ddapc_restraint_control)
            ddapc_restraint_control => dft_control%qs_control%ddapc_restraint_control(i)

            CALL cp_ddapc_apply_RS(qs_env, energy%ddapc_restraint(i), v_hartree_gspace, &
                                   v_spin_ddapc_rest_g, ddapc_restraint_control, calculate_forces)
         END DO
      END IF
      CALL cp_ddapc_apply_RF(qs_env, rho_tot_gspace, energy%hartree, v_hartree_gspace, &
                             calculate_forces, Itype_of_density="FULL DENSITY")

      ! CJM Copying the real-space Hartree potential to KS_ENV
      IF ((.NOT. just_energy) .OR. et_coupling_calc) THEN
         CALL pw_transfer(v_hartree_gspace, v_hartree_rspace)
         CALL pw_scale(v_hartree_rspace, v_hartree_rspace%pw_grid%dvol)
         IF (explicit_potential) THEN
            CALL pw_transfer(v_spin_ddapc_rest_g, v_spin_ddapc_rest_r)
            CALL pw_scale(v_spin_ddapc_rest_r, v_spin_ddapc_rest_r%pw_grid%dvol)
            IF (et_coupling_calc) THEN
               IF (qs_env%et_coupling%keep_matrix) THEN
                  IF (qs_env%et_coupling%first_run) THEN
                     NULLIFY (qs_env%et_coupling%rest_mat(1)%matrix)
                     ALLOCATE (qs_env%et_coupling%rest_mat(1)%matrix)
                     CALL dbcsr_copy(qs_env%et_coupling%rest_mat(1)%matrix, ks_matrix(1, 1)%matrix, &
                                     name="ET_RESTRAINT_MATRIX_B")
                     CALL dbcsr_set(qs_env%et_coupling%rest_mat(1)%matrix, 0.0_dp)
                     CALL integrate_v_rspace(v_spin_ddapc_rest_r, &
                                             hmat=qs_env%et_coupling%rest_mat(1), &
                                             qs_env=qs_env, calculate_forces=.FALSE.)
                     qs_env%et_coupling%order_p = &
                        dft_control%qs_control%ddapc_restraint_control(1)%ddapc_order_p
                     qs_env%et_coupling%e1 = dft_control%qs_control%ddapc_restraint_control(1)%strength
                     qs_env%et_coupling%keep_matrix = .FALSE.
                  ELSE
                     NULLIFY (qs_env%et_coupling%rest_mat(2)%matrix)
                     ALLOCATE (qs_env%et_coupling%rest_mat(2)%matrix)
                     CALL dbcsr_copy(qs_env%et_coupling%rest_mat(2)%matrix, ks_matrix(1, 1)%matrix, &
                                     name="ET_RESTRAINT_MATRIX_B")
                     CALL dbcsr_set(qs_env%et_coupling%rest_mat(2)%matrix, 0.0_dp)
                     CALL integrate_v_rspace(v_spin_ddapc_rest_r, &
                                             hmat=qs_env%et_coupling%rest_mat(2), &
                                             qs_env=qs_env, calculate_forces=.FALSE.)
                  END IF
               END IF
            END IF
         END IF
      END IF

      IF (explicit_potential) THEN
         CALL auxbas_pw_pool%give_back_pw(v_spin_ddapc_rest_g)
      END IF
      CALL timestop(handle)

   END SUBROUTINE qs_ks_ddapc

! **************************************************************************************************
!> \brief Device-resident Hartree potential with DDAPC decoupling (MULTIPOLE solver), step 1.
!>        Replaces calc_rho_tot_gspace and pw_poisson_solve: rho_tot = rho_core + rho_g and
!>        v_H = rho_tot*influence_fn stay on the GPU (same rounding as the host code). Only the
!>        low-|G| part of rho_tot_gspace needed by the DDAPC charge fit (|G| <= GCUT, with G=0) is
!>        written; qs_charges%total_rho_gspace is set as in calc_rho_tot_gspace. The caller
!>        guarantees decoupling only (no restraint, solvation, QM/MM), no forces or virial, no GAPW
!>        and one spin; the remaining conditions are checked here.
!> \param qs_env ...
!> \param poisson_env ...
!> \param rho_tot_gspace ...
!> \param done .FALSE.: the original code has to run (rho_tot_gspace may hold a partial result)
! **************************************************************************************************
   SUBROUTINE qs_ks_ddapc_device_begin(qs_env, poisson_env, rho_tot_gspace, done)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(pw_poisson_type), POINTER                     :: poisson_env
      TYPE(pw_c1d_gs_type), INTENT(INOUT)                :: rho_tot_gspace
      LOGICAL, INTENT(OUT)                               :: done

      CHARACTER(len=*), PARAMETER :: routineN = 'qs_ks_ddapc_device_begin'

      INTEGER                                            :: handle, ig, nprefix
      REAL(KIND=dp)                                      :: gcut, gcut2
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(pw_c1d_gs_type), DIMENSION(:), POINTER        :: rho_g
      TYPE(pw_c1d_gs_type), POINTER                      :: rho_core
      TYPE(pw_grid_type), POINTER                        :: grid
      TYPE(qs_charges_type), POINTER                     :: qs_charges
      TYPE(qs_rho_type), POINTER                         :: rho
      TYPE(section_vals_type), POINTER                   :: density_fit_section, input

      done = .FALSE.
      IF (.NOT. pw_gpu_res_enabled(pw_gpu_res_hartree_id)) RETURN
      IF (.NOT. ASSOCIATED(poisson_env)) RETURN
      IF (poisson_env%parameters%solver /= pw_poisson_multipole) RETURN
      NULLIFY (dft_control, rho_g, rho_core, qs_charges, rho, input, density_fit_section)
      CALL get_qs_env(qs_env, dft_control=dft_control, rho=rho, rho_core=rho_core, &
                      qs_charges=qs_charges, input=input)
      IF (.NOT. ASSOCIATED(rho) .OR. .NOT. ASSOCIATED(rho_core) .OR. .NOT. ASSOCIATED(qs_charges)) RETURN
      IF (dft_control%nspins /= 1) RETURN
      CALL qs_rho_get(rho, rho_g=rho_g)
      IF (.NOT. ASSOCIATED(rho_g)) RETURN
      IF (SIZE(rho_g) /= 1) RETURN
      grid => rho_tot_gspace%pw_grid
      IF (.NOT. ASSOCIATED(grid)) RETURN
      IF (grid%para%mode /= PW_MODE_LOCAL .OR. .NOT. grid%have_g0 .OR. grid%first_gne0 /= 2) RETURN
      IF (.NOT. ASSOCIATED(rho_core%pw_grid, grid) .OR. .NOT. ASSOCIATED(rho_g(1)%pw_grid, grid)) RETURN

      CALL timeset(routineN, handle)
      ! As in pw_poisson_solve; pw_transfer between identical grids is a copy.
      CALL pw_poisson_rebuild(poisson_env, rho_tot_gspace)
      IF (ASSOCIATED(poisson_env%green_fft)) THEN
         IF (poisson_env%green_fft%method == MULTIPOLE0D .AND. &
             ASSOCIATED(poisson_env%green_fft%influence_fn%pw_grid, grid) .AND. &
             ASSOCIATED(poisson_env%pw_pools(poisson_env%pw_level)%pool%pw_grid, grid)) THEN
            ! The charge fit reads rho_tot(1:igmax) (build_b_vector).
            density_fit_section => section_vals_get_subs_vals(input, "DFT%DENSITY_FITTING")
            CALL section_vals_val_get(density_fit_section, "GCUT", r_val=gcut)
            gcut2 = gcut*gcut
            nprefix = 1
            DO ig = grid%first_gne0, grid%ngpts_cut_local
               IF (grid%gsq(ig) > gcut2) EXIT
               nprefix = ig
            END DO
            done = pw_gpu_res_hartree_begin(rho_core, rho_g(1), poisson_env%green_fft%influence_fn, &
                                            rho_tot_gspace, nprefix)
         END IF
      END IF
      IF (done) qs_charges%total_rho_gspace = pw_integrate_function(rho_tot_gspace, isign=-1)
      CALL timestop(handle)
   END SUBROUTINE qs_ks_ddapc_device_begin

! **************************************************************************************************
!> \brief Device-resident Hartree potential with DDAPC decoupling, step 2. Replaces qs_ks_ddapc
!>        for decoupling only: charge fit on the host from the prefix of rho_tot_gspace, the
!>        correction of v_H, the decoupled Hartree energy and v_hartree_rspace = dvol*FFT^-1[v_H]
!>        on the GPU.
!> \param qs_env ...
!> \param rho_tot_gspace ...
!> \param energy ...
!> \param done .FALSE.: the original code has to run from calc_rho_tot_gspace on
! **************************************************************************************************
   SUBROUTINE qs_ks_ddapc_device_finish(qs_env, rho_tot_gspace, energy, done)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(pw_c1d_gs_type), INTENT(IN)                   :: rho_tot_gspace
      TYPE(qs_energy_type), POINTER                      :: energy
      LOGICAL, INTENT(OUT)                               :: done

      CHARACTER(len=*), PARAMETER :: routineN = 'qs_ks_ddapc_device_finish'

      INTEGER                                            :: handle, iw
      LOGICAL                                            :: ok
      REAL(KIND=dp)                                      :: dot, e_decpl
      REAL(KIND=dp), DIMENSION(:), POINTER               :: charges, radii
      TYPE(cp_logger_type), POINTER                      :: logger
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(pw_grid_type), POINTER                        :: grid
      TYPE(pw_r3d_rs_type), POINTER                      :: v_hartree_rspace
      TYPE(section_vals_type), POINTER                   :: density_fit_section, force_env_section, &
                                                            multipole_section, poisson_section

      CALL timeset(routineN, handle)
      CALL cite_reference(Blochl1995)
      NULLIFY (charges, radii, dft_control, particle_set, v_hartree_rspace, force_env_section)
      logger => cp_get_default_logger()
      CALL get_qs_env(qs_env, dft_control=dft_control, particle_set=particle_set, &
                      v_hartree_rspace=v_hartree_rspace, input=force_env_section)
      ! qs_ks_ddapc without restraints
      dft_control%qs_control%ddapc_explicit_potential = .FALSE.
      dft_control%qs_control%ddapc_restraint_is_spin = .FALSE.
      ! cp_ddapc_apply_CD with decoupling only
      poisson_section => section_vals_get_subs_vals(force_env_section, "DFT%POISSON")
      density_fit_section => section_vals_get_subs_vals(force_env_section, "DFT%DENSITY_FITTING")
      multipole_section => section_vals_get_subs_vals(poisson_section, "MULTIPOLE")
      iw = cp_print_key_unit_nr(logger, multipole_section, "PROGRAM_RUN_INFO", &
                                extension=".fitChargeLog")
      CALL get_ddapc(qs_env, .FALSE., density_fit_section, qout1=charges, out_radii=radii, &
                     ext_rho_tot_g=rho_tot_gspace, Itype_of_density="FULL DENSITY")
      IF (iw > 0) THEN
         e_decpl = 0.5_dp*DOT_PRODUCT(charges, MATMUL(qs_env%cp_ddapc_env%Md, charges))
         WRITE (iw, FMT="(T3,A,T60,F20.10)") "Decoupling Energy: ", e_decpl
      END IF
      ok = .FALSE.
      CALL modify_hartree_pot(rho_tot_gspace, density_fit_section, particle_set, &
                              qs_env%cp_ddapc_env%Mt, qs_env%cp_ddapc_env%AmI, radii, charges, &
                              device_done=ok)
      IF (ok) ok = pw_gpu_res_hartree_energy(dot)
      IF (ok) THEN
         ! energy = 0.5*pw_integral_ab(rho_tot_gspace, v_hartree_gspace)
         grid => rho_tot_gspace%pw_grid
         energy%hartree = 0.5_dp*(dot*grid%vol)
         ok = pw_gpu_res_hartree_finish(v_hartree_rspace, grid)
      END IF
      IF (.NOT. ok) CALL pw_gpu_res_hartree_cancel()
      DEALLOCATE (charges)
      DEALLOCATE (radii)
      CALL cp_print_key_finished_output(iw, logger, multipole_section, &
                                        "PROGRAM_RUN_INFO")
      done = ok
      CALL timestop(handle)
   END SUBROUTINE qs_ks_ddapc_device_finish

! **************************************************************************************************
!> \brief Routine to couple/decouple periodic images with the Bloechl scheme
!>
!>      The coupling/decoupling is obtaines evaluating terms E2 and E3 in
!>      J. Chem. Phys. Vol. 103 pp. 7422-7428.. The E2 terms is just a
!>      Ewald summation, and for performance reason I'm writing a specific
!>      driver instead of using and setting-up the environment of the already
!>      available routines
!> \param qs_env ...
!> \param rho_tot_gspace ...
!> \param energy ...
!> \param v_hartree_gspace ...
!> \param calculate_forces ...
!> \param Itype_of_density ...
!> \par History
!>      08.2005 created [tlaino]
!> \author Teodoro Laino
! **************************************************************************************************
   SUBROUTINE cp_ddapc_apply_CD(qs_env, rho_tot_gspace, energy, v_hartree_gspace, &
                                calculate_forces, Itype_of_density)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(pw_c1d_gs_type), INTENT(IN)                   :: rho_tot_gspace
      REAL(KIND=dp), INTENT(INOUT)                       :: energy
      TYPE(pw_c1d_gs_type), INTENT(IN)                   :: v_hartree_gspace
      LOGICAL, INTENT(IN), OPTIONAL                      :: calculate_forces
      CHARACTER(LEN=*)                                   :: Itype_of_density

      CHARACTER(len=*), PARAMETER                        :: routineN = 'cp_ddapc_apply_CD'

      INTEGER                                            :: handle, iw
      LOGICAL                                            :: apply_decpl, need_f
      REAL(KINd=dp)                                      :: e_decpl, e_recpl
      REAL(KIND=dp), DIMENSION(:), POINTER               :: charges, radii
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: dq
      TYPE(cell_type), POINTER                           :: cell, super_cell
      TYPE(cp_logger_type), POINTER                      :: logger
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(section_vals_type), POINTER                   :: density_fit_section, force_env_section, &
                                                            multipole_section, poisson_section, &
                                                            qmmm_periodic_section

      CALL timeset(routineN, handle)
      need_f = .FALSE.
      IF (PRESENT(calculate_forces)) need_f = calculate_forces
      logger => cp_get_default_logger()
      apply_decpl = qs_env%cp_ddapc_ewald%do_decoupling .OR. qs_env%cp_ddapc_ewald%do_qmmm_periodic_decpl
      IF (apply_decpl) THEN
         ! Initialize
         NULLIFY (multipole_section, &
                  poisson_section, &
                  force_env_section, &
                  particle_set, &
                  qmmm_periodic_section, &
                  density_fit_section, &
                  charges, &
                  radii, &
                  dq, &
                  cell, &
                  super_cell)

         CALL get_qs_env(qs_env=qs_env, &
                         input=force_env_section, &
                         particle_set=particle_set, &
                         cell=cell, &
                         super_cell=super_cell)
         CPASSERT(ASSOCIATED(qs_env%cp_ddapc_ewald))
         poisson_section => section_vals_get_subs_vals(force_env_section, "DFT%POISSON")

         density_fit_section => section_vals_get_subs_vals(force_env_section, "DFT%DENSITY_FITTING")

         IF (qs_env%cp_ddapc_ewald%do_decoupling) THEN
            multipole_section => section_vals_get_subs_vals(poisson_section, "MULTIPOLE")
         END IF
         IF (qs_env%cp_ddapc_ewald%do_qmmm_periodic_decpl) THEN
            qmmm_periodic_section => section_vals_get_subs_vals(force_env_section, "QMMM%PERIODIC")
            multipole_section => section_vals_get_subs_vals(qmmm_periodic_section, "MULTIPOLE")
         END IF
         ! Start the real calculation
         iw = cp_print_key_unit_nr(logger, multipole_section, "PROGRAM_RUN_INFO", &
                                   extension=".fitChargeLog")
         ! First we evaluate the charges at the corresponding SCF STEP
         IF (need_f) THEN
            CALL get_ddapc(qs_env, &
                           need_f, &
                           density_fit_section, &
                           qout1=charges, &
                           out_radii=radii, &
                           dq_out=dq, &
                           ext_rho_tot_g=rho_tot_gspace, &
                           Itype_of_density=Itype_of_density)
         ELSE
            CALL get_ddapc(qs_env, &
                           need_f, &
                           density_fit_section, &
                           qout1=charges, &
                           out_radii=radii, &
                           ext_rho_tot_g=rho_tot_gspace, &
                           Itype_of_density=Itype_of_density)
         END IF
         ! Evaluate the Ewald contribution to the decoupling/coupling E2 and E3
         IF (iw > 0) THEN
            e_decpl = 0.5_dp*DOT_PRODUCT(charges, MATMUL(qs_env%cp_ddapc_env%Md, charges))
            WRITE (iw, FMT="(T3,A,T60,F20.10)") "Decoupling Energy: ", e_decpl
         END IF
         IF (qs_env%cp_ddapc_ewald%do_qmmm_periodic_decpl .AND. (iw > 0)) THEN
            e_recpl = 0.5_dp*DOT_PRODUCT(charges, MATMUL(qs_env%cp_ddapc_env%Mr, charges))
            WRITE (iw, FMT="(T3,A,T60,F20.10)") "Recoupling Energy: ", e_recpl
         END IF
         CALL modify_hartree_pot(v_hartree_gspace, &
                                 density_fit_section, &
                                 particle_set, &
                                 qs_env%cp_ddapc_env%Mt, &
                                 qs_env%cp_ddapc_env%AmI, &
                                 radii, &
                                 charges)
         ! Modify the Hartree potential due to the decoupling/recoupling
         energy = 0.5_dp*pw_integral_ab(rho_tot_gspace, v_hartree_gspace)
         IF (need_f) THEN
            CALL ewald_ddapc_force(qs_env, qs_env%cp_ddapc_ewald%coeff_qm, &
                                   .FALSE., 1.0_dp, multipole_section, cell, particle_set, &
                                   radii, dq, charges)
            IF (qs_env%cp_ddapc_ewald%do_qmmm_periodic_decpl) THEN
               CALL ewald_ddapc_force(qs_env, qs_env%cp_ddapc_ewald%coeff_mm, &
                                      .TRUE., -1.0_dp, multipole_section, super_cell, particle_set, &
                                      radii, dq, charges)
            END IF
         END IF
         ! Clean the allocated arrays
         DEALLOCATE (charges)
         DEALLOCATE (radii)
         IF (ASSOCIATED(dq)) THEN
            DEALLOCATE (dq)
         END IF
         CALL cp_print_key_finished_output(iw, logger, multipole_section, &
                                           "PROGRAM_RUN_INFO")
      END IF
      CALL timestop(handle)
   END SUBROUTINE cp_ddapc_apply_CD

! **************************************************************************************************
!> \brief Routine to apply RESTRAINT/CONSTRAINTS to the density
!>      with the Bloechl scheme
!> \param qs_env ...
!> \param energy_res ...
!> \param v_hartree_gspace ...
!> \param v_spin_ddapc_rest_g ...
!> \param ddapc_restraint_control ...
!> \param calculate_forces ...
!> \par History
!>      08.2005 created [tlaino]
!> \author Teodoro Laino
! **************************************************************************************************
   SUBROUTINE cp_ddapc_apply_RS(qs_env, energy_res, v_hartree_gspace, &
                                v_spin_ddapc_rest_g, ddapc_restraint_control, calculate_forces)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      REAL(KIND=dp), INTENT(INOUT), OPTIONAL             :: energy_res
      TYPE(pw_c1d_gs_type), INTENT(IN)                   :: v_hartree_gspace, v_spin_ddapc_rest_g
      TYPE(ddapc_restraint_type), POINTER                :: ddapc_restraint_control
      LOGICAL, INTENT(IN), OPTIONAL                      :: calculate_forces

      CHARACTER(len=*), PARAMETER                        :: routineN = 'cp_ddapc_apply_RS'

      INTEGER                                            :: handle, iw, my_id
      LOGICAL                                            :: apply_restrain, need_f
      REAL(KIND=dp), DIMENSION(:), POINTER               :: charges, radii
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: dq
      TYPE(cell_type), POINTER                           :: cell, super_cell
      TYPE(cp_logger_type), POINTER                      :: logger
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(section_vals_type), POINTER                   :: density_fit_section, force_env_section, &
                                                            restraint_section

      CALL timeset(routineN, handle)
      NULLIFY (dft_control, restraint_section, force_env_section, particle_set, &
               charges, radii, dq, cell, density_fit_section, super_cell)
      need_f = .FALSE.

      CALL get_qs_env(qs_env=qs_env, &
                      input=force_env_section, &
                      particle_set=particle_set, &
                      cell=cell, &
                      super_cell=super_cell, &
                      dft_control=dft_control)

      IF (PRESENT(calculate_forces)) need_f = calculate_forces
      apply_restrain = dft_control%qs_control%ddapc_restraint
      logger => cp_get_default_logger()
      IF (apply_restrain) THEN
         ! Initialize
         density_fit_section => section_vals_get_subs_vals(force_env_section, "DFT%DENSITY_FITTING")
         restraint_section => section_vals_get_subs_vals(force_env_section, "DFT%QS%DDAPC_RESTRAINT")
         iw = cp_print_key_unit_nr(logger, restraint_section, "PROGRAM_RUN_INFO", &
                                   extension=".fitChargeLog")
         ! First we evaluate the charges at the corresponding SCF STEP
         my_id = ddapc_restraint_control%density_type
         IF (need_f) THEN
            CALL get_ddapc(qs_env, &
                           need_f, &
                           density_fit_section, &
                           density_type=my_id, &
                           qout1=charges, &
                           out_radii=radii, &
                           dq_out=dq, iwc=iw)
         ELSE
            CALL get_ddapc(qs_env, &
                           need_f, &
                           density_fit_section, &
                           density_type=my_id, &
                           qout1=charges, &
                           out_radii=radii, iwc=iw)
         END IF

         ! Modify the Hartree potential due to the restrain or the v_spin_ddapc_rest_g
         IF ((my_id == do_spin_density) .OR. dft_control%qs_control%et_coupling_calc) THEN
            CALL restraint_functional_potential(v_spin_ddapc_rest_g, density_fit_section, &
                                                particle_set, qs_env%cp_ddapc_env%AmI, radii, charges, &
                                                ddapc_restraint_control, energy_res)
         ELSE
            CALL restraint_functional_potential(v_hartree_gspace, density_fit_section, &
                                                particle_set, qs_env%cp_ddapc_env%AmI, radii, charges, &
                                                ddapc_restraint_control, energy_res)
         END IF

         IF (need_f) THEN
            CALL restraint_functional_force(qs_env, &
                                            ddapc_restraint_control, &
                                            dq, &
                                            charges, &
                                            SIZE(radii), &
                                            particle_set)
         END IF
         ! Clean the allocated arrays
         DEALLOCATE (charges)
         DEALLOCATE (radii)
         IF (ASSOCIATED(dq)) THEN
            DEALLOCATE (dq)
         END IF
         CALL cp_print_key_finished_output(iw, logger, restraint_section, &
                                           "PROGRAM_RUN_INFO")
      END IF
      CALL timestop(handle)
   END SUBROUTINE cp_ddapc_apply_RS

! **************************************************************************************************
!> \brief Routine to apply a reaction field during SCF (SCRF) with the Bloechl scheme
!> \param qs_env ...
!> \param rho_tot_gspace ...
!> \param energy ...
!> \param v_hartree_gspace ...
!> \param calculate_forces ...
!> \param Itype_of_density ...
!> \par History
!>      08.2005 created [tlaino]
!> \author Teodoro Laino
! **************************************************************************************************
   SUBROUTINE cp_ddapc_apply_RF(qs_env, rho_tot_gspace, energy, &
                                v_hartree_gspace, calculate_forces, Itype_of_density)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(pw_c1d_gs_type), INTENT(IN)                   :: rho_tot_gspace
      REAL(KIND=dp), INTENT(INOUT)                       :: energy
      TYPE(pw_c1d_gs_type), INTENT(IN)                   :: v_hartree_gspace
      LOGICAL, INTENT(IN), OPTIONAL                      :: calculate_forces
      CHARACTER(LEN=*)                                   :: Itype_of_density

      CHARACTER(len=*), PARAMETER                        :: routineN = 'cp_ddapc_apply_RF'

      INTEGER                                            :: handle, iw
      LOGICAL                                            :: apply_solvation, need_f
      REAL(KINd=dp)                                      :: e_recpl
      REAL(KIND=dp), DIMENSION(:), POINTER               :: charges, radii
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: dq
      TYPE(cell_type), POINTER                           :: cell, super_cell
      TYPE(cp_logger_type), POINTER                      :: logger
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(section_vals_type), POINTER                   :: density_fit_section, force_env_section, &
                                                            solvation_section

      CALL timeset(routineN, handle)
      need_f = .FALSE.
      IF (PRESENT(calculate_forces)) need_f = calculate_forces
      logger => cp_get_default_logger()
      apply_solvation = qs_env%cp_ddapc_ewald%do_solvation
      IF (apply_solvation) THEN
         ! Initialize
         NULLIFY (force_env_section, particle_set, charges, &
                  radii, dq, cell, super_cell)

         CALL get_qs_env(qs_env=qs_env, &
                         input=force_env_section, &
                         particle_set=particle_set, &
                         cell=cell, &
                         super_cell=super_cell)

         solvation_section => section_vals_get_subs_vals(force_env_section, "DFT%SCRF")
         ! Start the real calculation
         iw = cp_print_key_unit_nr(logger, solvation_section, "PROGRAM_RUN_INFO", &
                                   extension=".fitChargeLog")
         density_fit_section => section_vals_get_subs_vals(force_env_section, "DFT%DENSITY_FITTING")
         ! First we evaluate the charges at the corresponding SCF STEP
         IF (need_f) THEN
            CALL get_ddapc(qs_env, &
                           need_f, &
                           density_fit_section, &
                           qout1=charges, &
                           out_radii=radii, &
                           dq_out=dq, &
                           ext_rho_tot_g=rho_tot_gspace, &
                           Itype_of_density=Itype_of_density)
         ELSE
            CALL get_ddapc(qs_env, &
                           need_f, &
                           density_fit_section, &
                           qout1=charges, &
                           out_radii=radii, &
                           ext_rho_tot_g=rho_tot_gspace, &
                           Itype_of_density=Itype_of_density)
         END IF
         ! Evaluate the Ewald contribution to the decoupling/coupling E2 and E3
         IF (iw > 0) THEN
            e_recpl = 0.5_dp*DOT_PRODUCT(charges, MATMUL(qs_env%cp_ddapc_env%Ms, charges))
            WRITE (iw, FMT="(T3,A,T60,F20.10)") "Solvation  Energy: ", e_recpl
         END IF
         CALL modify_hartree_pot(v_hartree_gspace, &
                                 density_fit_section, &
                                 particle_set, &
                                 qs_env%cp_ddapc_env%Ms, &
                                 qs_env%cp_ddapc_env%AmI, &
                                 radii, &
                                 charges)
         ! Modify the Hartree potential due to the reaction field
         energy = 0.5_dp*pw_integral_ab(rho_tot_gspace, v_hartree_gspace)
         IF (need_f) THEN
            CALL solvation_ddapc_force(qs_env, solvation_section, particle_set, &
                                       radii, dq, charges)
         END IF
         ! Clean the allocated arrays
         DEALLOCATE (charges)
         DEALLOCATE (radii)
         IF (ASSOCIATED(dq)) THEN
            DEALLOCATE (dq)
         END IF
         CALL cp_print_key_finished_output(iw, logger, solvation_section, &
                                           "PROGRAM_RUN_INFO")
      END IF
      CALL timestop(handle)
   END SUBROUTINE cp_ddapc_apply_RF

END MODULE cp_ddapc
