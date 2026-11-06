!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Transfers densities from PW to RS grids and potentials from PW to RS
!> \par History
!>      - Copied from qs_coolocate_Density and qs_integrate_potenntial
!> \author JGH (04.2014)
! **************************************************************************************************
MODULE rs_pw_interface
   USE ISO_C_BINDING,                   ONLY: C_ASSOCIATED, C_NULL_PTR, C_PTR, C_SIZE_T
   USE cp_log_handling,                 ONLY: cp_to_string
   USE cp_spline_utils,                 ONLY: pw_interp,&
                                              pw_prolongate_s3,&
                                              pw_restrict_s3,&
                                              spline3_pbc_interp
   USE gaussian_gridlevels,             ONLY: gridlevel_info_type
   USE input_section_types,             ONLY: section_vals_val_get
   USE kinds,                           ONLY: dp
   USE offload_api,                     ONLY: offload_get_buffer_device_pointer,&
                                              offload_grid_bind,&
                                              offload_grid_device,&
                                              offload_grid_host,&
                                              offload_grid_prepare
   USE pw_env_types,                    ONLY: pw_env_get,&
                                              pw_env_type
   USE pw_gpu,                          ONLY: pw_gpu_res_density,&
                                              pw_gpu_res_density_grid,&
                                              pw_gpu_res_density_id,&
                                              pw_gpu_res_enabled,&
                                              pw_gpu_res_ensure_host,&
                                              pw_gpu_res_potential
   USE pw_methods,                      ONLY: pw_axpy,&
                                              pw_copy,&
                                              pw_transfer,&
                                              pw_zero
   USE pw_grid_types,                   ONLY: PW_MODE_LOCAL, pw_grid_type
   USE pw_pool_types,                   ONLY: pw_pool_p_type,&
                                              pw_pools_create_pws,&
                                              pw_pools_give_back_pws
   USE pw_types,                        ONLY: pw_c1d_gs_type,&
                                              pw_r3d_rs_type
   USE realspace_grid_types,            ONLY: realspace_grid_desc_p_type,&
                                              realspace_grid_type,&
                                              transfer_pw2rs,&
                                              transfer_rs2pw
#include "../base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'rs_pw_interface'
! *** Public subroutines ***

   PUBLIC :: density_rs2pw, &
             potential_pw2rs, rs_pw_grid_prepare

CONTAINS

! **************************************************************************************************
!> \brief Only serial, unpadded grids share the PW FFT layout without communication.
   FUNCTION grid_layout_ok(rs, grid) RESULT(ok)
      TYPE(realspace_grid_type), INTENT(IN) :: rs
      TYPE(pw_grid_type), POINTER :: grid
      LOGICAL :: ok

      ok = .FALSE.
      IF (.NOT. ASSOCIATED(rs%desc) .OR. .NOT. ASSOCIATED(grid)) RETURN
      IF (.NOT. ASSOCIATED(rs%desc%pw, grid)) RETURN
      IF (rs%desc%distributed .OR. rs%desc%parallel .OR. rs%desc%border /= 0) RETURN
      IF (grid%para%mode /= PW_MODE_LOCAL .OR. grid%spherical) RETURN
      IF (ANY(rs%npts_local /= grid%npts)) RETURN
      IF (INT(rs%ngpts_local, C_SIZE_T) /= PRODUCT(INT(grid%npts, C_SIZE_T))) RETURN
      IF (ANY(rs%lb_local /= grid%bounds(1, :)) .OR. ANY(rs%ub_local /= grid%bounds(2, :))) RETURN
      IF (.NOT. ASSOCIATED(rs%buffer%host_buffer)) RETURN
      IF (SIZE(rs%buffer%host_buffer) < rs%ngpts_local) RETURN
      ok = .TRUE.
   END FUNCTION grid_layout_ok

! **************************************************************************************************
!> \brief Opt in only the collocation immediately followed by density_rs2pw.
   SUBROUTINE rs_pw_grid_prepare(rs_grids)
      TYPE(realspace_grid_type), DIMENSION(:), INTENT(IN) :: rs_grids
      INTEGER :: l
      LOGICAL :: defer

      DO l = 1, SIZE(rs_grids)
         defer = .FALSE.
         IF (ASSOCIATED(rs_grids(l)%desc)) THEN
            defer = pw_gpu_res_enabled(pw_gpu_res_density_id) .AND. &
                    grid_layout_ok(rs_grids(l), rs_grids(l)%desc%pw)
         END IF
         CALL offload_grid_prepare(rs_grids(l)%buffer, defer)
      END DO
   END SUBROUTINE rs_pw_grid_prepare

! **************************************************************************************************
!> \brief given partial densities on the realspace multigrids,
!>      computes the full density on the plane wave grids, both in real and
!>      gspace
!> \param pw_env ...
!> \param rs_rho ...
!> \param rho ...
!> \param rho_gspace ...
!> \param defer_host Keep electron densities on device until a host consumer requests them.
!> \note
!>      should contain all communication in the collocation of the density
!>      in the case of replicated grids
! **************************************************************************************************
   SUBROUTINE density_rs2pw(pw_env, rs_rho, rho, rho_gspace, defer_host)

      TYPE(pw_env_type), INTENT(IN)                      :: pw_env
      TYPE(realspace_grid_type), DIMENSION(:), &
         INTENT(IN)                                      :: rs_rho
      TYPE(pw_r3d_rs_type), INTENT(INOUT)                :: rho
      TYPE(pw_c1d_gs_type), INTENT(INOUT)                :: rho_gspace
      LOGICAL, INTENT(IN), OPTIONAL                     :: defer_host

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'density_rs2pw'

      INTEGER                                            :: handle, igrid_level, interp_kind
      LOGICAL                                            :: lazy, grid_done, any_device
      TYPE(C_PTR)                                        :: rs_dev_ptrs(SIZE(rs_rho))
      TYPE(gridlevel_info_type), POINTER                 :: gridlevel_info
      TYPE(pw_c1d_gs_type), ALLOCATABLE, DIMENSION(:)    :: mgrid_gspace
      TYPE(pw_pool_p_type), DIMENSION(:), POINTER        :: pw_pools
      TYPE(pw_r3d_rs_type), ALLOCATABLE, DIMENSION(:)    :: mgrid_rspace
      TYPE(realspace_grid_desc_p_type), DIMENSION(:), &
         POINTER                                         :: rs_descs

      CALL timeset(routineN, handle)
      lazy = .FALSE.
      IF (PRESENT(defer_host)) lazy = defer_host
      NULLIFY (gridlevel_info, rs_descs, pw_pools)
      CALL pw_env_get(pw_env, rs_descs=rs_descs, pw_pools=pw_pools)

      gridlevel_info => pw_env%gridlevel_info

      CALL section_vals_val_get(pw_env%interp_section, "KIND", i_val=interp_kind)

      CALL pw_pools_create_pws(pw_pools, mgrid_rspace)

      CALL pw_pools_create_pws(pw_pools, mgrid_gspace)

      rs_dev_ptrs = C_NULL_PTR
      grid_done = .FALSE.
      any_device = .FALSE.
      IF (interp_kind == pw_interp) THEN
         DO igrid_level = 1, gridlevel_info%ngrid_levels
            IF (grid_layout_ok(rs_rho(igrid_level), mgrid_gspace(igrid_level)%pw_grid)) THEN
               rs_dev_ptrs(igrid_level) = offload_grid_device(rs_rho(igrid_level)%buffer, &
                                                             rs_rho(igrid_level)%ngpts_local)
               any_device = any_device .OR. C_ASSOCIATED(rs_dev_ptrs(igrid_level))
            END IF
         END DO
         IF (any_device) THEN
            DO igrid_level = 1, gridlevel_info%ngrid_levels
               IF (C_ASSOCIATED(rs_dev_ptrs(igrid_level))) CYCLE
               CALL offload_grid_host(rs_rho(igrid_level)%buffer)
               CALL transfer_rs2pw(rs_rho(igrid_level), mgrid_rspace(igrid_level))
            END DO
            grid_done = pw_gpu_res_density_grid(rs_dev_ptrs(1:gridlevel_info%ngrid_levels), &
                                                mgrid_gspace(1:gridlevel_info%ngrid_levels), rho, rho_gspace, &
                                                mgrid_rspace(1:gridlevel_info%ngrid_levels))
         END IF
      END IF
      DO igrid_level = 1, gridlevel_info%ngrid_levels
         IF (.NOT. grid_done) CALL offload_grid_host(rs_rho(igrid_level)%buffer)
         CALL offload_grid_prepare(rs_rho(igrid_level)%buffer, .FALSE.)
      END DO

      IF (grid_done) THEN
         ! The Grid allocations were read directly; all PW outputs are complete.
      ELSE IF (gridlevel_info%ngrid_levels == 1) THEN
         CALL transfer_rs2pw(rs_rho(1), rho)
         CALL pw_transfer(rho, rho_gspace)
         IF (rho%pw_grid%spherical) THEN ! rho_gspace = rho
            CALL pw_transfer(rho_gspace, rho)
         END IF
      ELSE
         DO igrid_level = 1, gridlevel_info%ngrid_levels
            CALL transfer_rs2pw(rs_rho(igrid_level), &
                                mgrid_rspace(igrid_level))
         END DO

         ! we want both rho and rho_gspace, the latter for Hartree and co-workers.
         SELECT CASE (interp_kind)
         CASE (pw_interp)
            ! The same sequence on the GPU, without the host passes; rho/rho_gspace stay resident.
            IF (.NOT. pw_gpu_res_density(mgrid_rspace(1:gridlevel_info%ngrid_levels), &
                                         mgrid_gspace(1:gridlevel_info%ngrid_levels), rho, rho_gspace)) THEN
               CALL pw_zero(rho_gspace)
               DO igrid_level = 1, gridlevel_info%ngrid_levels
                  CALL pw_transfer(mgrid_rspace(igrid_level), &
                                   mgrid_gspace(igrid_level))
                  CALL pw_axpy(mgrid_gspace(igrid_level), rho_gspace)
               END DO
               CALL pw_transfer(rho_gspace, rho)
            END IF
         CASE (spline3_pbc_interp)
            DO igrid_level = gridlevel_info%ngrid_levels, 2, -1
               CALL pw_prolongate_s3(mgrid_rspace(igrid_level), &
                                     mgrid_rspace(igrid_level - 1), pw_pools(igrid_level)%pool, &
                                     pw_env%interp_section)
            END DO
            CALL pw_copy(mgrid_rspace(1), rho)
            CALL pw_transfer(rho, rho_gspace)
         CASE default
            CALL cp_abort(__LOCATION__, &
                          "interpolator "// &
                          cp_to_string(interp_kind))
         END SELECT
      END IF

      ! Derivative and response callers keep the original host-visible contract.
      IF (.NOT. lazy) THEN
         CALL pw_gpu_res_ensure_host(rho)
         CALL pw_gpu_res_ensure_host(rho_gspace)
      END IF

      ! *** give back the pw multi-grids
      CALL pw_pools_give_back_pws(pw_pools, mgrid_gspace)
      CALL pw_pools_give_back_pws(pw_pools, mgrid_rspace)
      CALL timestop(handle)

   END SUBROUTINE density_rs2pw

! **************************************************************************************************
!> \brief transfers a potential from a pw_grid to a vector of
!>      realspace multigrids
!> \param rs_v OUTPUT: the potential on the realspace multigrids
!> \param v_rspace INPUT : the potential on a planewave grid in Rspace
!> \param pw_env ...
!> \param device_grid The caller integrates these grids on the device: levels whose layout matches
!>        their PW grid receive the potential in their own device memory (no host copy).
!> \par History
!>      09.2006 created [Joost VandeVondele]
!> \note
!>      extracted from integrate_v_rspace
!>      should contain all parallel communication of integrate_v_rspace in the
!>      case of replicated grids.
! **************************************************************************************************
   SUBROUTINE potential_pw2rs(rs_v, v_rspace, pw_env, device_grid)

      TYPE(realspace_grid_type), DIMENSION(:), &
         INTENT(IN)                                      :: rs_v
      TYPE(pw_r3d_rs_type), INTENT(IN)                   :: v_rspace
      TYPE(pw_env_type), INTENT(IN)                      :: pw_env
      LOGICAL, INTENT(IN), OPTIONAL                      :: device_grid

      CHARACTER(len=*), PARAMETER                        :: routineN = 'potential_pw2rs'

      INTEGER                                            :: auxbas_grid, handle, igrid_level, &
                                                            interp_kind
      REAL(KIND=dp)                                      :: scale
      LOGICAL                                            :: any_device, bind_grid, done
      TYPE(C_PTR), DIMENSION(SIZE(rs_v))                 :: rs_dev
      TYPE(gridlevel_info_type), POINTER                 :: gridlevel_info
      TYPE(pw_c1d_gs_type), ALLOCATABLE, DIMENSION(:)    :: mgrid_gspace
      TYPE(pw_pool_p_type), DIMENSION(:), POINTER        :: pw_pools
      TYPE(pw_r3d_rs_type), ALLOCATABLE, DIMENSION(:)    :: mgrid_rspace

      CALL timeset(routineN, handle)

      ! *** set up of the potential on the multigrids
      CALL pw_env_get(pw_env, pw_pools=pw_pools, gridlevel_info=gridlevel_info, &
                      auxbas_grid=auxbas_grid)

      CALL pw_pools_create_pws(pw_pools, mgrid_rspace)

      ! use either realspace or fft techniques to get the potential on the rs multigrids
      CALL section_vals_val_get(pw_env%interp_section, "KIND", i_val=interp_kind)
      rs_dev = C_NULL_PTR
      any_device = .FALSE.
      bind_grid = .FALSE.
      IF (PRESENT(device_grid)) bind_grid = device_grid
      bind_grid = bind_grid .AND. interp_kind == pw_interp
      DO igrid_level = 1, gridlevel_info%ngrid_levels
         CALL offload_grid_prepare(rs_v(igrid_level)%buffer, .FALSE.)
         ! Levels with the FFT layout of their PW grid are written in their own device memory.
         IF (bind_grid) THEN
            IF (grid_layout_ok(rs_v(igrid_level), mgrid_rspace(igrid_level)%pw_grid)) THEN
               rs_dev(igrid_level) = offload_get_buffer_device_pointer(rs_v(igrid_level)%buffer)
               IF (C_ASSOCIATED(rs_dev(igrid_level))) any_device = .TRUE.
            END IF
         END IF
      END DO
      SELECT CASE (interp_kind)
      CASE (pw_interp)
         ! The same sequence on the GPU; the G-space multigrids never leave the device.
         done = .FALSE.
         IF (any_device) THEN
            done = pw_gpu_res_potential(v_rspace, mgrid_rspace(1:gridlevel_info%ngrid_levels), auxbas_grid, &
                                        rs_dev(1:gridlevel_info%ngrid_levels))
         ELSE
            done = pw_gpu_res_potential(v_rspace, mgrid_rspace(1:gridlevel_info%ngrid_levels), auxbas_grid)
         END IF
         IF (.NOT. done) THEN
            rs_dev = C_NULL_PTR
            CALL pw_gpu_res_ensure_host(v_rspace)
            CALL pw_pools_create_pws(pw_pools, mgrid_gspace)
            CALL pw_transfer(v_rspace, mgrid_gspace(auxbas_grid))
            DO igrid_level = 1, gridlevel_info%ngrid_levels
               IF (igrid_level /= auxbas_grid) THEN
                  CALL pw_copy(mgrid_gspace(auxbas_grid), mgrid_gspace(igrid_level))
                  CALL pw_transfer(mgrid_gspace(igrid_level), mgrid_rspace(igrid_level))
               ELSE
                  IF (mgrid_gspace(auxbas_grid)%pw_grid%spherical) THEN
                     CALL pw_transfer(mgrid_gspace(auxbas_grid), mgrid_rspace(auxbas_grid))
                  ELSE ! fft forward + backward should be identical
                     CALL pw_copy(v_rspace, mgrid_rspace(auxbas_grid))
                  END IF
               END IF
               ! *** Multiply by the grid volume element ratio ***
               IF (igrid_level /= auxbas_grid) THEN
                  scale = mgrid_rspace(igrid_level)%pw_grid%dvol/ &
                          mgrid_rspace(auxbas_grid)%pw_grid%dvol
                  mgrid_rspace(igrid_level)%array = &
                     scale*mgrid_rspace(igrid_level)%array
               END IF
            END DO
            CALL pw_pools_give_back_pws(pw_pools, mgrid_gspace)
         END IF
      CASE (spline3_pbc_interp)
         CALL pw_copy(v_rspace, mgrid_rspace(1))
         DO igrid_level = 1, gridlevel_info%ngrid_levels - 1
            CALL pw_zero(mgrid_rspace(igrid_level + 1))
            CALL pw_restrict_s3(mgrid_rspace(igrid_level), &
                                mgrid_rspace(igrid_level + 1), pw_pools(igrid_level + 1)%pool, &
                                pw_env%interp_section)
            ! *** Multiply by the grid volume element ratio
            mgrid_rspace(igrid_level + 1)%array = &
               mgrid_rspace(igrid_level + 1)%array*8._dp
         END DO
      CASE default
         CALL cp_abort(__LOCATION__, &
                       "interpolation not supported "// &
                       cp_to_string(interp_kind))
      END SELECT

      DO igrid_level = 1, gridlevel_info%ngrid_levels
         IF (C_ASSOCIATED(rs_dev(igrid_level))) THEN
            done = offload_grid_bind(rs_v(igrid_level)%buffer, rs_dev(igrid_level), rs_v(igrid_level)%ngpts_local)
            CPASSERT(done)
            CYCLE
         END IF
         CALL transfer_pw2rs(rs_v(igrid_level), &
                             mgrid_rspace(igrid_level))
      END DO
      ! *** give back the pw multi-grids
      CALL pw_pools_give_back_pws(pw_pools, mgrid_rspace)

      CALL timestop(handle)

   END SUBROUTINE potential_pw2rs

END MODULE rs_pw_interface
