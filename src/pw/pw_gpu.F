!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \note
!> This module contains routines necessary to operate on plane waves on GPUs
!     >  independently of the GPU platform.
!> \par History
!>      BGL (06-Mar-2008)  : Created
!>      AG  (18-May-2012)  : Refacturing:
!>                           - added explicit interfaces to C routines
!>                           - enable double precision complex transformations
!>      AG  (11-Sept-2012) : Modifications:
!>                          - use pointers if precision mapping is not required
!>                          - use OMP for mapping
!>      MT  (Jan 2022)     : Modifications
!>                          - use a generic interface for fft calls to GPUs
!>                          - Support both Nvidia and AMD GPUs. Other GPUs manufacturers
!>                            can be added easily.
!> \author Benjamin G. Levine
! **************************************************************************************************
MODULE pw_gpu
   USE ISO_C_BINDING,                   ONLY: C_ASSOCIATED,&
                                              C_DOUBLE,&
                                              C_INT,&
                                              C_LOC,&
                                              C_NULL_PTR,&
                                              C_PTR,&
                                              C_SIZE_T
   USE cp_log_handling,                 ONLY: cp_logger_get_default_io_unit
   USE fft_tools,                       ONLY: &
        cube_transpose_1, cube_transpose_2, fft_scratch_sizes, fft_scratch_type, get_fft_scratch, &
        release_fft_scratch, x_to_yz, xz_to_yz, yz_to_x, yz_to_xz
   USE kinds,                           ONLY: dp
   USE mathconstants,                   ONLY: z_zero
   USE message_passing,                 ONLY: mp_cart_type
   USE pw_grid_types,                   ONLY: FULLSPACE,&
                                              PW_MODE_LOCAL,&
                                              pw_grid_type
   USE pw_types,                        ONLY: pw_c1d_gs_type,&
                                              pw_r3d_rs_type
#include "../base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: pw_gpu_r3dc1d_3d
   PUBLIC :: pw_gpu_c1dr3d_3d
   PUBLIC :: pw_gpu_r3dc1d_3d_ps
   PUBLIC :: pw_gpu_c1dr3d_3d_ps
   PUBLIC :: pw_gpu_init, pw_gpu_finalize
   PUBLIC :: pw_gpu_set_memory_limit, pw_gpu_set_host_fallback, pw_gpu_get_allocated_bytes
   PUBLIC :: pw_gpu_fused_xc_available
   PUBLIC :: pw_gpu_xc_full_pipeline_available, pw_gpu_xc_full_pipeline
   PUBLIC :: pw_gpu_gradient_c1d_r3d
   PUBLIC :: pw_gpu_divergence_r3d
   PUBLIC :: pw_gpu_filter_r3d
   PUBLIC :: pw_gpu_copy, pw_gpu_axpy, pw_gpu_remap_copy, pw_gpu_remap_axpy
   PUBLIC :: modify_hartree_pot_gpu, pbe_lda_eval_gpu
   PUBLIC :: pw_gpu_res_enabled, pw_gpu_res_invalidate
   PUBLIC :: pw_gpu_res_density, pw_gpu_res_potential
   PUBLIC :: pw_gpu_res_density_grid
   PUBLIC :: pw_gpu_res_integrate_r, pw_gpu_res_ensure_host, pw_gpu_res_ensure_host_ptr
   PUBLIC :: pw_gpu_res_hartree_begin, pw_gpu_res_hartree_modify, pw_gpu_res_hartree_energy, &
             pw_gpu_res_hartree_finish, pw_gpu_res_hartree_cancel
   PUBLIC :: pw_gpu_res_vtot_prepare, pw_gpu_res_vtot_commit
   PUBLIC :: pw_gpu_res_vtot_skip_host
   PUBLIC :: pw_gpu_res_set_lazy, pw_gpu_res_defer_density, pw_gpu_res_discard
   PUBLIC :: pw_gpu_res_density_id, pw_gpu_res_hartree_id, pw_gpu_res_xcin_id, pw_gpu_res_pot_id

   ! Residency features, see pw_gpu_res_enabled.
   INTEGER, PARAMETER :: pw_gpu_res_density_id = 1, pw_gpu_res_hartree_id = 2, &
                         pw_gpu_res_xcin_id = 3, pw_gpu_res_pot_id = 4

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'pw_gpu'
   LOGICAL, PARAMETER, PRIVATE :: debug_this_module = .FALSE.

   INTERFACE pw_gpu_res_ensure_host
      MODULE PROCEDURE pw_gpu_res_ensure_host_r, pw_gpu_res_ensure_host_g
   END INTERFACE

CONTAINS

! **************************************************************************************************
!> \brief Try the nonzero-G Hartree correction. G=0 remains in the caller.
! **************************************************************************************************
   LOGICAL FUNCTION modify_hartree_pot_gpu(v, g, gsq, r, radii, cv, n, na, nr, gcut2, vol) RESULT(done)
      INTEGER, INTENT(IN) :: n, na, nr
      COMPLEX(KIND=dp), TARGET, INTENT(INOUT) :: v(n)
      REAL(KIND=dp), TARGET, INTENT(IN) :: g(3, n), gsq(n), r(3, na), radii(nr), cv(nr*na)
      REAL(KIND=dp), INTENT(IN) :: gcut2, vol
      INTERFACE
         FUNCTION modify_hartree_pot_gpu_c(v, g, gsq, r, radii, cv, n, na, nr, gcut2, vol) &
            BIND(C, name="modify_hartree_pot_gpu_c") RESULT(res)
            IMPORT :: C_PTR, C_SIZE_T, C_DOUBLE, C_INT
            TYPE(C_PTR), VALUE :: v, g, gsq, r, radii, cv
            INTEGER(KIND=C_SIZE_T), VALUE :: n, na, nr
            REAL(KIND=C_DOUBLE), VALUE :: gcut2, vol
            INTEGER(KIND=C_INT) :: res
         END FUNCTION modify_hartree_pot_gpu_c
      END INTERFACE

      done = .FALSE.
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      done = modify_hartree_pot_gpu_c(C_LOC(v), C_LOC(g), C_LOC(gsq), C_LOC(r), C_LOC(radii), C_LOC(cv), &
                                    INT(n, C_SIZE_T), INT(na, C_SIZE_T), INT(nr, C_SIZE_T), gcut2, vol) /= 0_C_INT
#endif
   END FUNCTION modify_hartree_pot_gpu

! **************************************************************************************************
!> \brief Try unpolarized PBE energy and first derivatives, preserving existing accumulated values.
! **************************************************************************************************
   LOGICAL FUNCTION pbe_lda_eval_gpu(rho, norm, e0, erho, enorm, n, grad, epsilon, &
                                   kappa, beta, mu, sc, sx) RESULT(done)
      INTEGER, INTENT(IN) :: n, grad
      REAL(KIND=dp), TARGET, INTENT(IN) :: rho(n), norm(n)
      REAL(KIND=dp), TARGET, INTENT(INOUT) :: e0(n), erho(n), enorm(n)
      REAL(KIND=dp), INTENT(IN) :: epsilon, kappa, beta, mu, sc, sx
      INTERFACE
         FUNCTION pbe_lda_eval_gpu_c(rho, norm, e0, erho, enorm, n, grad, epsilon, &
                                     kappa, beta, mu, sc, sx) BIND(C, name="pbe_lda_eval_gpu_c") RESULT(res)
            IMPORT :: C_PTR, C_SIZE_T, C_DOUBLE, C_INT
            TYPE(C_PTR), VALUE :: rho, norm, e0, erho, enorm
            INTEGER(KIND=C_SIZE_T), VALUE :: n
            INTEGER(KIND=C_INT), VALUE :: grad
            REAL(KIND=C_DOUBLE), VALUE :: epsilon, kappa, beta, mu, sc, sx
            INTEGER(KIND=C_INT) :: res
         END FUNCTION pbe_lda_eval_gpu_c
      END INTERFACE

      done = .FALSE.
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      done = pbe_lda_eval_gpu_c(C_LOC(rho), C_LOC(norm), C_LOC(e0), C_LOC(erho), C_LOC(enorm), &
                              INT(n, C_SIZE_T), INT(grad, C_INT), epsilon, kappa, beta, mu, sc, sx) /= 0_C_INT
#endif
   END FUNCTION pbe_lda_eval_gpu

! **************************************************************************************************
!> \brief Try a synchronous GPU grid operation; false leaves outputs unchanged.
! **************************************************************************************************
   LOGICAL FUNCTION pw_gpu_copy(x, y, nx, ny, cx, cy, alpha) RESULT(done)
      TYPE(C_PTR), VALUE :: x, y
      INTEGER(KIND=C_SIZE_T), INTENT(IN) :: nx, ny
      INTEGER, INTENT(IN) :: cx, cy
      REAL(KIND=dp), INTENT(IN) :: alpha
      INTERFACE
         FUNCTION pw_gpu_copy_c(x, y, nx, ny, cx, cy, alpha) &
            BIND(C, name="pw_gpu_copy_c") RESULT(res)
            IMPORT :: C_PTR, C_INT, C_SIZE_T, C_DOUBLE
            TYPE(C_PTR), VALUE :: x, y
            INTEGER(KIND=C_SIZE_T), VALUE :: nx, ny
            INTEGER(KIND=C_INT), VALUE :: cx, cy
            REAL(KIND=C_DOUBLE), VALUE :: alpha
            INTEGER(KIND=C_INT) :: res
         END FUNCTION pw_gpu_copy_c
      END INTERFACE

      done = .FALSE.
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      done = pw_gpu_copy_c(x, y, nx, ny, INT(cx, C_INT), INT(cy, C_INT), &
         alpha) /= 0_C_INT
#endif
   END FUNCTION pw_gpu_copy

! **************************************************************************************************
!> \brief Try a synchronous GPU grid operation; false leaves outputs unchanged.
! **************************************************************************************************
   LOGICAL FUNCTION pw_gpu_axpy(x, y, nx, ny, cx, cy, alpha, beta) RESULT(done)
      TYPE(C_PTR), VALUE :: x, y
      INTEGER(KIND=C_SIZE_T), INTENT(IN) :: nx, ny
      INTEGER, INTENT(IN) :: cx, cy
      REAL(KIND=dp), INTENT(IN) :: alpha, beta
      INTERFACE
         FUNCTION pw_gpu_axpy_c(x, y, nx, ny, cx, cy, alpha, beta) &
            BIND(C, name="pw_gpu_axpy_c") RESULT(res)
            IMPORT :: C_PTR, C_INT, C_SIZE_T, C_DOUBLE
            TYPE(C_PTR), VALUE :: x, y
            INTEGER(KIND=C_SIZE_T), VALUE :: nx, ny
            INTEGER(KIND=C_INT), VALUE :: cx, cy
            REAL(KIND=C_DOUBLE), VALUE :: alpha, beta
            INTEGER(KIND=C_INT) :: res
         END FUNCTION pw_gpu_axpy_c
      END INTERFACE

      done = .FALSE.
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      done = pw_gpu_axpy_c(x, y, nx, ny, INT(cx, C_INT), INT(cy, C_INT), &
         alpha, beta) /= 0_C_INT
#endif
   END FUNCTION pw_gpu_axpy

! **************************************************************************************************
!> \brief Try a synchronous GPU grid operation; false leaves outputs unchanged.
! **************************************************************************************************
   LOGICAL FUNCTION pw_gpu_remap_copy(x, y, nx, ny, cx, cy, gidx, scatter, alpha) RESULT(done)
      TYPE(C_PTR), VALUE :: x, y
      INTEGER(KIND=C_SIZE_T), INTENT(IN) :: nx, ny
      INTEGER, INTENT(IN) :: cx, cy
      INTEGER(KIND=C_INT), CONTIGUOUS, TARGET, INTENT(IN) :: gidx(:)
      LOGICAL, INTENT(IN) :: scatter
      REAL(KIND=dp), INTENT(IN) :: alpha
      INTERFACE
         FUNCTION pw_gpu_remap_copy_c(x, y, nx, ny, cx, cy, map, direction, alpha) &
            BIND(C, name="pw_gpu_remap_copy_c") RESULT(res)
            IMPORT :: C_PTR, C_INT, C_SIZE_T, C_DOUBLE
            TYPE(C_PTR), VALUE :: x, y, map
            INTEGER(KIND=C_SIZE_T), VALUE :: nx, ny
            INTEGER(KIND=C_INT), VALUE :: cx, cy, direction
            REAL(KIND=C_DOUBLE), VALUE :: alpha
            INTEGER(KIND=C_INT) :: res
         END FUNCTION pw_gpu_remap_copy_c
      END INTERFACE

      done = .FALSE.
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      done = pw_gpu_remap_copy_c(x, y, nx, ny, INT(cx, C_INT), INT(cy, C_INT), &
         C_LOC(gidx), MERGE(1_C_INT, 0_C_INT, scatter), alpha) /= 0_C_INT
#endif
   END FUNCTION pw_gpu_remap_copy

! **************************************************************************************************
!> \brief Try a synchronous GPU grid operation; false leaves outputs unchanged.
! **************************************************************************************************
   LOGICAL FUNCTION pw_gpu_remap_axpy(x, y, nx, ny, cx, cy, gidx, scatter, alpha, beta) RESULT(done)
      TYPE(C_PTR), VALUE :: x, y
      INTEGER(KIND=C_SIZE_T), INTENT(IN) :: nx, ny
      INTEGER, INTENT(IN) :: cx, cy
      INTEGER(KIND=C_INT), CONTIGUOUS, TARGET, INTENT(IN) :: gidx(:)
      LOGICAL, INTENT(IN) :: scatter
      REAL(KIND=dp), INTENT(IN) :: alpha, beta
      INTERFACE
         FUNCTION pw_gpu_remap_axpy_c(x, y, nx, ny, cx, cy, map, direction, alpha, beta) &
            BIND(C, name="pw_gpu_remap_axpy_c") RESULT(res)
            IMPORT :: C_PTR, C_INT, C_SIZE_T, C_DOUBLE
            TYPE(C_PTR), VALUE :: x, y, map
            INTEGER(KIND=C_SIZE_T), VALUE :: nx, ny
            INTEGER(KIND=C_INT), VALUE :: cx, cy, direction
            REAL(KIND=C_DOUBLE), VALUE :: alpha, beta
            INTEGER(KIND=C_INT) :: res
         END FUNCTION pw_gpu_remap_axpy_c
      END INTERFACE

      done = .FALSE.
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      done = pw_gpu_remap_axpy_c(x, y, nx, ny, INT(cx, C_INT), INT(cy, C_INT), &
         C_LOC(gidx), MERGE(1_C_INT, 0_C_INT, scatter), alpha, beta) /= 0_C_INT
#endif
   END FUNCTION pw_gpu_remap_axpy


! **************************************************************************************************
!> \brief Sets the per-rank budget in MiB (zero means unlimited). Environment overrides take precedence.
!> \param limit_mb Memory budget.
! **************************************************************************************************
   SUBROUTINE pw_gpu_set_memory_limit(limit_mb)
      INTEGER, INTENT(IN)                                :: limit_mb

      INTERFACE
         SUBROUTINE pw_gpu_set_memory_limit_c(limit_mb) BIND(C, name="pw_gpu_set_memory_limit")
            IMPORT :: C_INT
            INTEGER(KIND=C_INT), VALUE                   :: limit_mb
         END SUBROUTINE pw_gpu_set_memory_limit_c
      END INTERFACE

      CPASSERT(limit_mb >= 0)
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      CALL pw_gpu_set_memory_limit_c(INT(limit_mb, KIND=C_INT))
#else
      MARK_USED(limit_mb)
#endif
   END SUBROUTINE pw_gpu_set_memory_limit

! **************************************************************************************************
!> \brief Enables CPU processing when the plane-wave device memory budget is exhausted.
!> \param enable Whether host fallback is allowed.
! **************************************************************************************************
   SUBROUTINE pw_gpu_set_host_fallback(enable)
      LOGICAL, INTENT(IN)                                :: enable

      INTERFACE
         SUBROUTINE pw_gpu_set_host_fallback_c(enable) BIND(C, name="pw_gpu_set_host_fallback")
            IMPORT :: C_INT
            INTEGER(KIND=C_INT), VALUE                   :: enable
         END SUBROUTINE pw_gpu_set_host_fallback_c
      END INTERFACE

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      CALL pw_gpu_set_host_fallback_c(MERGE(1_C_INT, 0_C_INT, enable))
#else
      MARK_USED(enable)
#endif
   END SUBROUTINE pw_gpu_set_host_fallback

! **************************************************************************************************
!> \brief Queries explicitly allocated plane-wave device buffers, excluding opaque FFT workspaces.
!> \param allocated_bytes Current allocation.
!> \param peak_bytes Peak allocation since GPU initialization.
! **************************************************************************************************
   SUBROUTINE pw_gpu_get_allocated_bytes(allocated_bytes, peak_bytes)
      INTEGER(KIND=C_SIZE_T), INTENT(OUT)                :: allocated_bytes, peak_bytes

      INTERFACE
         SUBROUTINE pw_gpu_get_allocated_bytes_c(allocated_bytes, peak_bytes) &
            BIND(C, name="pw_gpu_get_allocated_bytes")
            IMPORT :: C_SIZE_T
            INTEGER(KIND=C_SIZE_T), INTENT(OUT)          :: allocated_bytes, peak_bytes
         END SUBROUTINE pw_gpu_get_allocated_bytes_c
      END INTERFACE

      allocated_bytes = 0_C_SIZE_T
      peak_bytes = 0_C_SIZE_T
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      CALL pw_gpu_get_allocated_bytes_c(allocated_bytes, peak_bytes)
#endif
   END SUBROUTINE pw_gpu_get_allocated_bytes

! **************************************************************************************************
!> \brief Allocates resources on the gpu device for gpu fft acceleration
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE pw_gpu_init()
      INTEGER                                            :: dummy, opts, unit_nr
      INTERFACE
         SUBROUTINE pw_gpu_init_c() BIND(C, name="pw_gpu_init")
         END SUBROUTINE pw_gpu_init_c
      END INTERFACE
      INTERFACE
         FUNCTION pw_gpu_get_options_c() BIND(C, name="pw_gpu_get_options") RESULT(res)
            IMPORT :: C_INT
            INTEGER(KIND=C_INT)                          :: res
         END FUNCTION pw_gpu_get_options_c
      END INTERFACE

      MARK_USED(dummy) ! TODO: fix fpretty
      opts = 0
      unit_nr = cp_logger_get_default_io_unit()
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      CALL pw_gpu_init_c()
      opts = INT(pw_gpu_get_options_c())
      IF (unit_nr > 0) THEN
         WRITE (unit_nr, '(T2,A,T41,A,L1,A,L1)') "PW_GPU| resident-XC patch options:", &
            "MAP_CACHE=", BTEST(opts, 0), "  FUSED_XC=", BTEST(opts, 1)
         WRITE (unit_nr, '(T2,A,T41,A,L1,A,L1,A,L1,A,L1)') "PW_GPU| residency (stage 1):", &
            "DENSITY=", BTEST(opts, 2), " HARTREE=", BTEST(opts, 3), " XCIN=", BTEST(opts, 4), &
            " POT=", BTEST(opts, 5)
      END IF
#else
      ! Nothing to do.
      MARK_USED(opts)
      MARK_USED(unit_nr)
#endif
   END SUBROUTINE pw_gpu_init

! **************************************************************************************************
!> \brief Releases resources on the gpu device for gpu fft acceleration
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE pw_gpu_finalize()
      INTEGER                                            :: dummy
      INTERFACE
         SUBROUTINE pw_gpu_finalize_c() BIND(C, name="pw_gpu_finalize")
         END SUBROUTINE pw_gpu_finalize_c
      END INTERFACE

      MARK_USED(dummy) ! TODO: fix fpretty
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      CALL pw_gpu_finalize_c()
#else
      ! Nothing to do.
#endif
   END SUBROUTINE pw_gpu_finalize

! **************************************************************************************************
!> \brief perform an fft followed by a gather on the gpu
!> \param pw1 ...
!> \param pw2 ...
!> \author Benjamin G Levine
! **************************************************************************************************
   SUBROUTINE pw_gpu_r3dc1d_3d(pw1, pw2)
      TYPE(pw_r3d_rs_type), INTENT(IN)                   :: pw1
      TYPE(pw_c1d_gs_type), INTENT(INOUT)                :: pw2

      CHARACTER(len=*), PARAMETER                        :: routineN = 'pw_gpu_r3dc1d_3d'

      COMPLEX(KIND=dp), POINTER                          :: ptr_pwout
      INTEGER                                            :: handle, l1, l2, l3, ngpts, nmaps
      INTEGER, DIMENSION(:), POINTER                     :: npts
      INTEGER, POINTER                                   :: ptr_ghatmap
      REAL(KIND=dp)                                      :: scale
      REAL(KIND=dp), POINTER                             :: ptr_pwin
      INTERFACE
         SUBROUTINE pw_gpu_cfffg_c(din, zout, ghatmap, npts, ngpts, nmaps, scale, grid_id) &
            BIND(C, name="pw_gpu_cfffg_cached")
            IMPORT
            TYPE(C_PTR), INTENT(IN), VALUE               :: din
            TYPE(C_PTR), VALUE                           :: zout
            TYPE(C_PTR), INTENT(IN), VALUE               :: ghatmap
            INTEGER(KIND=C_INT), DIMENSION(*), INTENT(IN):: npts
            INTEGER(KIND=C_INT), INTENT(IN), VALUE       :: ngpts, nmaps
            REAL(KIND=C_DOUBLE), INTENT(IN), VALUE       :: scale
            INTEGER(KIND=C_INT), INTENT(IN), VALUE       :: grid_id

         END SUBROUTINE pw_gpu_cfffg_c
      END INTERFACE

      CALL timeset(routineN, handle)

      scale = 1.0_dp/REAL(pw1%pw_grid%ngpts, KIND=dp)

      ngpts = SIZE(pw2%pw_grid%gsq)
      l1 = LBOUND(pw1%array, 1)
      l2 = LBOUND(pw1%array, 2)
      l3 = LBOUND(pw1%array, 3)
      npts => pw1%pw_grid%npts

      ! pointers to data arrays
      ptr_pwin => pw1%array(l1, l2, l3)
      ptr_pwout => pw2%array(1)

      ! pointer to map array
      nmaps = SIZE(pw2%pw_grid%g_hatmap, 2)
      ptr_ghatmap => pw2%pw_grid%g_hatmap(1, 1)

      ! invoke the combined transformation (the grid id keys the device-side map cache)
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      CALL pw_gpu_cfffg_c(c_loc(ptr_pwin), c_loc(ptr_pwout), c_loc(ptr_ghatmap), npts, ngpts, nmaps, &
                          scale, pw2%pw_grid%id_nr)
#else
      CPABORT("Compiled without pw offloading.")
#endif

      CALL timestop(handle)
   END SUBROUTINE pw_gpu_r3dc1d_3d

! **************************************************************************************************
!> \brief perform an scatter followed by a fft on the gpu
!> \param pw1 ...
!> \param pw2 ...
!> \author Benjamin G Levine
! **************************************************************************************************
   SUBROUTINE pw_gpu_c1dr3d_3d(pw1, pw2)
      TYPE(pw_c1d_gs_type), INTENT(IN)                   :: pw1
      TYPE(pw_r3d_rs_type), INTENT(INOUT)                :: pw2

      CHARACTER(len=*), PARAMETER                        :: routineN = 'pw_gpu_c1dr3d_3d'

      COMPLEX(KIND=dp), POINTER                          :: ptr_pwin
      INTEGER                                            :: handle, l1, l2, l3, ngpts, nmaps
      INTEGER, DIMENSION(:), POINTER                     :: npts
      INTEGER, POINTER                                   :: ptr_ghatmap
      REAL(KIND=dp)                                      :: scale
      REAL(KIND=dp), POINTER                             :: ptr_pwout
      INTERFACE
         SUBROUTINE pw_gpu_sfffc_c(zin, dout, ghatmap, npts, ngpts, nmaps, scale, grid_id) &
            BIND(C, name="pw_gpu_sfffc_cached")
            IMPORT
            TYPE(C_PTR), INTENT(IN), VALUE               :: zin
            TYPE(C_PTR), VALUE                           :: dout
            TYPE(C_PTR), INTENT(IN), VALUE               :: ghatmap
            INTEGER(KIND=C_INT), DIMENSION(*), INTENT(IN):: npts
            INTEGER(KIND=C_INT), INTENT(IN), VALUE       :: ngpts, nmaps
            REAL(KIND=C_DOUBLE), INTENT(IN), VALUE       :: scale
            INTEGER(KIND=C_INT), INTENT(IN), VALUE       :: grid_id
         END SUBROUTINE pw_gpu_sfffc_c
      END INTERFACE

      CALL timeset(routineN, handle)

      scale = 1.0_dp

      ngpts = SIZE(pw1%pw_grid%gsq)
      l1 = LBOUND(pw2%array, 1)
      l2 = LBOUND(pw2%array, 2)
      l3 = LBOUND(pw2%array, 3)
      npts => pw1%pw_grid%npts

      ! pointers to data arrays
      ptr_pwin => pw1%array(1)
      ptr_pwout => pw2%array(l1, l2, l3)

      ! pointer to map array
      nmaps = SIZE(pw1%pw_grid%g_hatmap, 2)
      ptr_ghatmap => pw1%pw_grid%g_hatmap(1, 1)

      ! invoke the combined transformation
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      CALL pw_gpu_sfffc_c(c_loc(ptr_pwin), c_loc(ptr_pwout), c_loc(ptr_ghatmap), npts, ngpts, nmaps, &
                          scale, pw1%pw_grid%id_nr)
#else
      CPABORT("Compiled without pw offloading")
#endif

      CALL timestop(handle)
   END SUBROUTINE pw_gpu_c1dr3d_3d

! **************************************************************************************************
!> \brief perform an parallel fft followed by a gather on the gpu
!> \param pw1 ...
!> \param pw2 ...
!> \author Andreas Gloess
! **************************************************************************************************
   SUBROUTINE pw_gpu_r3dc1d_3d_ps(pw1, pw2)
      TYPE(pw_r3d_rs_type), INTENT(IN)                   :: pw1
      TYPE(pw_c1d_gs_type), INTENT(INOUT)                :: pw2

      CHARACTER(len=*), PARAMETER :: routineN = 'pw_gpu_r3dc1d_3d_ps'

      COMPLEX(KIND=dp), DIMENSION(:, :), POINTER         :: grays, pbuf, qbuf, rbuf, sbuf
      COMPLEX(KIND=dp), DIMENSION(:, :, :), POINTER      :: tbuf
      INTEGER                                            :: g_pos, handle, lg, lmax, mg, mmax, mx2, &
                                                            mz2, n1, n2, ngpts, nmax, numtask, rp
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: p2p
      INTEGER, DIMENSION(2)                              :: r_dim, r_pos
      INTEGER, DIMENSION(:), POINTER                     :: n, nloc, nyzray
      INTEGER, DIMENSION(:, :, :, :), POINTER            :: bo
      REAL(KIND=dp)                                      :: scale
      TYPE(fft_scratch_sizes)                            :: fft_scratch_size
      TYPE(fft_scratch_type), POINTER                    :: fft_scratch
      TYPE(mp_cart_type)                                 :: rs_group

      CALL timeset(routineN, handle)

      scale = 1.0_dp/REAL(pw1%pw_grid%ngpts, KIND=dp)

      ! dimensions
      n => pw1%pw_grid%npts
      nloc => pw1%pw_grid%npts_local
      grays => pw1%pw_grid%grays
      ngpts = nloc(1)*nloc(2)*nloc(3)

      !..transform
      IF (pw1%pw_grid%para%ray_distribution) THEN
         rs_group = pw1%pw_grid%para%group
         nyzray => pw1%pw_grid%para%nyzray
         bo => pw1%pw_grid%para%bo

         g_pos = rs_group%mepos
         numtask = rs_group%num_pe
         r_dim = rs_group%num_pe_cart
         r_pos = rs_group%mepos_cart

         lg = SIZE(grays, 1)
         mg = SIZE(grays, 2)
         mmax = MAX(mg, 1)
         lmax = MAX(lg, (ngpts/mmax + 1))

         ALLOCATE (p2p(0:numtask - 1))

         CALL rs_group%rank_compare(rs_group, p2p)

         rp = p2p(g_pos)
         mx2 = bo(2, 1, rp, 2) - bo(1, 1, rp, 2) + 1
         mz2 = bo(2, 3, rp, 2) - bo(1, 3, rp, 2) + 1
         n1 = MAXVAL(bo(2, 1, :, 1) - bo(1, 1, :, 1) + 1)
         n2 = MAXVAL(bo(2, 2, :, 1) - bo(1, 2, :, 1) + 1)
         nmax = MAX((2*n2)/numtask, 2)*mx2*mz2
         nmax = MAX(nmax, n1*MAXVAL(nyzray))

         fft_scratch_size%nx = nloc(1)
         fft_scratch_size%ny = nloc(2)
         fft_scratch_size%nz = nloc(3)
         fft_scratch_size%lmax = lmax
         fft_scratch_size%mmax = mmax
         fft_scratch_size%mx1 = bo(2, 1, rp, 1) - bo(1, 1, rp, 1) + 1
         fft_scratch_size%mx2 = mx2
         fft_scratch_size%my1 = bo(2, 2, rp, 1) - bo(1, 2, rp, 1) + 1
         fft_scratch_size%mz2 = mz2
         fft_scratch_size%lg = lg
         fft_scratch_size%mg = mg
         fft_scratch_size%nbx = MAXVAL(bo(2, 1, :, 2))
         fft_scratch_size%nbz = MAXVAL(bo(2, 3, :, 2))
         fft_scratch_size%mcz1 = MAXVAL(bo(2, 3, :, 1) - bo(1, 3, :, 1) + 1)
         fft_scratch_size%mcx2 = MAXVAL(bo(2, 1, :, 2) - bo(1, 1, :, 2) + 1)
         fft_scratch_size%mcz2 = MAXVAL(bo(2, 3, :, 2) - bo(1, 3, :, 2) + 1)
         fft_scratch_size%nmax = nmax
         fft_scratch_size%nmray = MAXVAL(nyzray)
         fft_scratch_size%nyzray = nyzray(g_pos)
         fft_scratch_size%rs_group = rs_group
         fft_scratch_size%g_pos = g_pos
         fft_scratch_size%r_pos = r_pos
         fft_scratch_size%r_dim = r_dim
         fft_scratch_size%numtask = numtask

         IF (r_dim(2) > 1) THEN
            !
            ! real space is distributed over x and y coordinate
            ! we have two stages of communication
            !
            IF (r_dim(1) == 1) &
               CPABORT("This processor distribution is not supported.")

            CALL get_fft_scratch(fft_scratch, tf_type=300, n=n, fft_sizes=fft_scratch_size)

            ! assign buffers
            qbuf => fft_scratch%p2buf
            rbuf => fft_scratch%p3buf
            pbuf => fft_scratch%p4buf
            sbuf => fft_scratch%p5buf

            ! FFT along z
            CALL pw_gpu_cf(pw1, qbuf)

            ! Exchange data ( transpose of matrix )
            CALL cube_transpose_2(qbuf, bo(:, :, :, 1), bo(:, :, :, 2), rbuf, fft_scratch)

            ! FFT along y
            ! use the inbuild fft-lib
            ! CALL fft_1dm(fft_scratch%fft_plan(2), rbuf, pbuf, 1.0_dp, stat)
            ! or cufft (works faster, but is only faster if plans are stored)
            CALL pw_gpu_f(rbuf, pbuf, +1, n(2), mx2*mz2)

            ! Exchange data ( transpose of matrix ) and sort
            CALL xz_to_yz(pbuf, rs_group, r_dim, g_pos, p2p, pw1%pw_grid%para%yzp, nyzray, &
                          bo(:, :, :, 2), sbuf, fft_scratch)

            ! FFT along x
            CALL pw_gpu_fg(sbuf, pw2, scale)

            CALL release_fft_scratch(fft_scratch)

         ELSE
            !
            ! real space is only distributed over x coordinate
            ! we have one stage of communication, after the transform of
            ! direction x
            !

            CALL get_fft_scratch(fft_scratch, tf_type=200, n=n, fft_sizes=fft_scratch_size)

            ! assign buffers
            tbuf => fft_scratch%tbuf
            sbuf => fft_scratch%r1buf

            ! FFT along y and z
            CALL pw_gpu_cff(pw1, tbuf)

            ! Exchange data ( transpose of matrix ) and sort
            CALL yz_to_x(tbuf, rs_group, g_pos, p2p, pw1%pw_grid%para%yzp, nyzray, &
                         bo(:, :, :, 2), sbuf, fft_scratch)

            ! FFT along x
            CALL pw_gpu_fg(sbuf, pw2, scale)

            CALL release_fft_scratch(fft_scratch)

         END IF

         DEALLOCATE (p2p)

!--------------------------------------------------------------------------
      ELSE
         CPABORT("Not implemented (no ray_distr.) in: pw_gpu_r3dc1d_3d_ps.")
      END IF

      CALL timestop(handle)
   END SUBROUTINE pw_gpu_r3dc1d_3d_ps

! **************************************************************************************************
!> \brief perform an parallel scatter followed by a fft on the gpu
!> \param pw1 ...
!> \param pw2 ...
!> \author Andreas Gloess
! **************************************************************************************************
   SUBROUTINE pw_gpu_c1dr3d_3d_ps(pw1, pw2)
      TYPE(pw_c1d_gs_type), INTENT(IN)                   :: pw1
      TYPE(pw_r3d_rs_type), INTENT(INOUT)                :: pw2

      CHARACTER(len=*), PARAMETER :: routineN = 'pw_gpu_c1dr3d_3d_ps'

      COMPLEX(KIND=dp), DIMENSION(:, :), POINTER         :: grays, pbuf, qbuf, rbuf, sbuf
      COMPLEX(KIND=dp), DIMENSION(:, :, :), POINTER      :: tbuf
      INTEGER                                            :: g_pos, handle, lg, lmax, mg, mmax, mx2, &
                                                            mz2, n1, n2, ngpts, nmax, numtask, rp
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: p2p
      INTEGER, DIMENSION(2)                              :: r_dim, r_pos
      INTEGER, DIMENSION(:), POINTER                     :: n, nloc, nyzray
      INTEGER, DIMENSION(:, :, :, :), POINTER            :: bo
      REAL(KIND=dp)                                      :: scale
      TYPE(fft_scratch_sizes)                            :: fft_scratch_size
      TYPE(fft_scratch_type), POINTER                    :: fft_scratch
      TYPE(mp_cart_type)                                 :: rs_group

      CALL timeset(routineN, handle)

      scale = 1.0_dp

      ! dimensions
      n => pw1%pw_grid%npts
      nloc => pw1%pw_grid%npts_local
      grays => pw1%pw_grid%grays
      ngpts = nloc(1)*nloc(2)*nloc(3)

      !..transform
      IF (pw1%pw_grid%para%ray_distribution) THEN
         rs_group = pw1%pw_grid%para%group
         nyzray => pw1%pw_grid%para%nyzray
         bo => pw1%pw_grid%para%bo

         g_pos = rs_group%mepos
         numtask = rs_group%num_pe
         r_dim = rs_group%num_pe_cart
         r_pos = rs_group%mepos_cart

         lg = SIZE(grays, 1)
         mg = SIZE(grays, 2)
         mmax = MAX(mg, 1)
         lmax = MAX(lg, (ngpts/mmax + 1))

         ALLOCATE (p2p(0:numtask - 1))

         CALL rs_group%rank_compare(rs_group, p2p)

         rp = p2p(g_pos)
         mx2 = bo(2, 1, rp, 2) - bo(1, 1, rp, 2) + 1
         mz2 = bo(2, 3, rp, 2) - bo(1, 3, rp, 2) + 1
         n1 = MAXVAL(bo(2, 1, :, 1) - bo(1, 1, :, 1) + 1)
         n2 = MAXVAL(bo(2, 2, :, 1) - bo(1, 2, :, 1) + 1)
         nmax = MAX((2*n2)/numtask, 2)*mx2*mz2
         nmax = MAX(nmax, n1*MAXVAL(nyzray))

         fft_scratch_size%nx = nloc(1)
         fft_scratch_size%ny = nloc(2)
         fft_scratch_size%nz = nloc(3)
         fft_scratch_size%lmax = lmax
         fft_scratch_size%mmax = mmax
         fft_scratch_size%mx1 = bo(2, 1, rp, 1) - bo(1, 1, rp, 1) + 1
         fft_scratch_size%mx2 = mx2
         fft_scratch_size%my1 = bo(2, 2, rp, 1) - bo(1, 2, rp, 1) + 1
         fft_scratch_size%mz2 = mz2
         fft_scratch_size%lg = lg
         fft_scratch_size%mg = mg
         fft_scratch_size%nbx = MAXVAL(bo(2, 1, :, 2))
         fft_scratch_size%nbz = MAXVAL(bo(2, 3, :, 2))
         fft_scratch_size%mcz1 = MAXVAL(bo(2, 3, :, 1) - bo(1, 3, :, 1) + 1)
         fft_scratch_size%mcx2 = MAXVAL(bo(2, 1, :, 2) - bo(1, 1, :, 2) + 1)
         fft_scratch_size%mcz2 = MAXVAL(bo(2, 3, :, 2) - bo(1, 3, :, 2) + 1)
         fft_scratch_size%nmax = nmax
         fft_scratch_size%nmray = MAXVAL(nyzray)
         fft_scratch_size%nyzray = nyzray(g_pos)
         fft_scratch_size%rs_group = rs_group
         fft_scratch_size%g_pos = g_pos
         fft_scratch_size%r_pos = r_pos
         fft_scratch_size%r_dim = r_dim
         fft_scratch_size%numtask = numtask

         IF (r_dim(2) > 1) THEN
            !
            ! real space is distributed over x and y coordinate
            ! we have two stages of communication
            !
            IF (r_dim(1) == 1) &
               CPABORT("This processor distribution is not supported.")

            CALL get_fft_scratch(fft_scratch, tf_type=300, n=n, fft_sizes=fft_scratch_size)

            ! assign buffers
            pbuf => fft_scratch%p7buf
            qbuf => fft_scratch%p4buf
            rbuf => fft_scratch%p3buf
            sbuf => fft_scratch%p2buf

            ! FFT along x
            CALL pw_gpu_sf(pw1, pbuf, scale)

            ! Exchange data ( transpose of matrix ) and sort
            IF (pw1%pw_grid%grid_span /= FULLSPACE) qbuf = z_zero
            CALL yz_to_xz(pbuf, rs_group, r_dim, g_pos, p2p, pw1%pw_grid%para%yzp, nyzray, &
                          bo(:, :, :, 2), qbuf, fft_scratch)

            ! FFT along y
            ! use the inbuild fft-lib
            ! CALL fft_1dm(fft_scratch%fft_plan(5), qbuf, rbuf, 1.0_dp, stat)
            ! or cufft (works faster, but is only faster if plans are stored)
            CALL pw_gpu_f(qbuf, rbuf, -1, n(2), mx2*mz2)

            ! Exchange data ( transpose of matrix )
            IF (pw1%pw_grid%grid_span /= FULLSPACE) sbuf = z_zero

            CALL cube_transpose_1(rbuf, bo(:, :, :, 2), bo(:, :, :, 1), sbuf, fft_scratch)

            ! FFT along z
            CALL pw_gpu_fc(sbuf, pw2)

            CALL release_fft_scratch(fft_scratch)

         ELSE
            !
            ! real space is only distributed over x coordinate
            ! we have one stage of communication, after the transform of
            ! direction x
            !

            CALL get_fft_scratch(fft_scratch, tf_type=200, n=n, fft_sizes=fft_scratch_size)

            ! assign buffers
            sbuf => fft_scratch%r1buf
            tbuf => fft_scratch%tbuf

            ! FFT along x
            CALL pw_gpu_sf(pw1, sbuf, scale)

            ! Exchange data ( transpose of matrix ) and sort
            IF (pw1%pw_grid%grid_span /= FULLSPACE) tbuf = z_zero
            CALL x_to_yz(sbuf, rs_group, g_pos, p2p, pw1%pw_grid%para%yzp, nyzray, &
                         bo(:, :, :, 2), tbuf, fft_scratch)

            ! FFT along y and z
            CALL pw_gpu_ffc(tbuf, pw2)

            CALL release_fft_scratch(fft_scratch)

         END IF

         DEALLOCATE (p2p)

!--------------------------------------------------------------------------
      ELSE
         CPABORT("Not implemented (no ray_distr.) in: pw_gpu_c1dr3d_3d_ps.")
      END IF

      CALL timestop(handle)
   END SUBROUTINE pw_gpu_c1dr3d_3d_ps

! **************************************************************************************************
!> \brief perform a parallel real_to_complex copy followed by a 2D-FFT on the gpu
!> \param pw1 ...
!> \param pwbuf ...
!> \author Andreas Gloess
! **************************************************************************************************
   SUBROUTINE pw_gpu_cff(pw1, pwbuf)
      TYPE(pw_r3d_rs_type), INTENT(IN)                   :: pw1
      COMPLEX(KIND=dp), DIMENSION(:, :, :), &
         INTENT(INOUT), TARGET                           :: pwbuf

      CHARACTER(len=*), PARAMETER                        :: routineN = 'pw_gpu_cff'

      COMPLEX(KIND=dp), POINTER                          :: ptr_pwout
      INTEGER                                            :: handle, l1, l2, l3
      INTEGER, DIMENSION(:), POINTER                     :: npts
      REAL(KIND=dp), POINTER                             :: ptr_pwin
      INTERFACE
         SUBROUTINE pw_gpu_cff_c(din, zout, npts) BIND(C, name="pw_gpu_cff")
            IMPORT
            TYPE(C_PTR), INTENT(IN), VALUE               :: din
            TYPE(C_PTR), VALUE                           :: zout
            INTEGER(KIND=C_INT), DIMENSION(*), INTENT(IN):: npts
         END SUBROUTINE pw_gpu_cff_c
      END INTERFACE

      CALL timeset(routineN, handle)

      ! dimensions
      npts => pw1%pw_grid%npts_local
      l1 = LBOUND(pw1%array, 1)
      l2 = LBOUND(pw1%array, 2)
      l3 = LBOUND(pw1%array, 3)

      ! pointers to data arrays
      ptr_pwin => pw1%array(l1, l2, l3)
      ptr_pwout => pwbuf(1, 1, 1)

      ! invoke the combined transformation
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      CALL pw_gpu_cff_c(c_loc(ptr_pwin), c_loc(ptr_pwout), npts)
#else
      CPABORT("Compiled without pw offloading")
#endif

      CALL timestop(handle)
   END SUBROUTINE pw_gpu_cff

! **************************************************************************************************
!> \brief perform a parallel 2D-FFT followed by a complex_to_real copy on the gpu
!> \param pwbuf ...
!> \param pw2 ...
!> \author Andreas Gloess
! **************************************************************************************************
   SUBROUTINE pw_gpu_ffc(pwbuf, pw2)
      COMPLEX(KIND=dp), DIMENSION(:, :, :), INTENT(IN), &
         TARGET                                          :: pwbuf
      TYPE(pw_r3d_rs_type), INTENT(IN)                   :: pw2

      CHARACTER(len=*), PARAMETER                        :: routineN = 'pw_gpu_ffc'

      COMPLEX(KIND=dp), POINTER                          :: ptr_pwin
      INTEGER                                            :: handle, l1, l2, l3
      INTEGER, DIMENSION(:), POINTER                     :: npts
      REAL(KIND=dp), POINTER                             :: ptr_pwout
      INTERFACE
         SUBROUTINE pw_gpu_ffc_c(zin, dout, npts) BIND(C, name="pw_gpu_ffc")
            IMPORT
            TYPE(C_PTR), INTENT(IN), VALUE               :: zin
            TYPE(C_PTR), VALUE                           :: dout
            INTEGER(KIND=C_INT), DIMENSION(*), INTENT(IN):: npts
         END SUBROUTINE pw_gpu_ffc_c
      END INTERFACE

      CALL timeset(routineN, handle)

      ! dimensions
      npts => pw2%pw_grid%npts_local
      l1 = LBOUND(pw2%array, 1)
      l2 = LBOUND(pw2%array, 2)
      l3 = LBOUND(pw2%array, 3)

      ! pointers to data arrays
      ptr_pwin => pwbuf(1, 1, 1)
      ptr_pwout => pw2%array(l1, l2, l3)

      ! invoke the combined transformation
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      CALL pw_gpu_ffc_c(c_loc(ptr_pwin), c_loc(ptr_pwout), npts)
#else
      CPABORT("Compiled without pw offloading")
#endif

      CALL timestop(handle)
   END SUBROUTINE pw_gpu_ffc

! **************************************************************************************************
!> \brief perform a parallel real_to_complex copy followed by a 1D-FFT on the gpu
!> \param pw1 ...
!> \param pwbuf ...
!> \author Andreas Gloess
! **************************************************************************************************
   SUBROUTINE pw_gpu_cf(pw1, pwbuf)
      TYPE(pw_r3d_rs_type), INTENT(IN)                   :: pw1
      COMPLEX(KIND=dp), DIMENSION(:, :), INTENT(INOUT), &
         TARGET                                          :: pwbuf

      CHARACTER(len=*), PARAMETER                        :: routineN = 'pw_gpu_cf'

      COMPLEX(KIND=dp), POINTER                          :: ptr_pwout
      INTEGER                                            :: handle, l1, l2, l3
      INTEGER, DIMENSION(:), POINTER                     :: npts
      REAL(KIND=dp), POINTER                             :: ptr_pwin
      INTERFACE
         SUBROUTINE pw_gpu_cf_c(din, zout, npts) BIND(C, name="pw_gpu_cf")
            IMPORT
            TYPE(C_PTR), INTENT(IN), VALUE               :: din
            TYPE(C_PTR), VALUE                           :: zout
            INTEGER(KIND=C_INT), DIMENSION(*), INTENT(IN):: npts
         END SUBROUTINE pw_gpu_cf_c
      END INTERFACE

      CALL timeset(routineN, handle)

      ! dimensions
      npts => pw1%pw_grid%npts_local
      l1 = LBOUND(pw1%array, 1)
      l2 = LBOUND(pw1%array, 2)
      l3 = LBOUND(pw1%array, 3)

      ! pointers to data arrays
      ptr_pwin => pw1%array(l1, l2, l3)
      ptr_pwout => pwbuf(1, 1)

      ! invoke the combined transformation
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      CALL pw_gpu_cf_c(c_loc(ptr_pwin), c_loc(ptr_pwout), npts)
#else
      CPABORT("Compiled without pw offloading")
#endif
      CALL timestop(handle)
   END SUBROUTINE pw_gpu_cf

! **************************************************************************************************
!> \brief perform a parallel 1D-FFT followed by a complex_to_real copy on the gpu
!> \param pwbuf ...
!> \param pw2 ...
!> \author Andreas Gloess
! **************************************************************************************************
   SUBROUTINE pw_gpu_fc(pwbuf, pw2)
      COMPLEX(KIND=dp), DIMENSION(:, :), INTENT(IN), &
         TARGET                                          :: pwbuf
      TYPE(pw_r3d_rs_type), INTENT(IN)                   :: pw2

      CHARACTER(len=*), PARAMETER                        :: routineN = 'pw_gpu_fc'

      COMPLEX(KIND=dp), POINTER                          :: ptr_pwin
      INTEGER                                            :: handle, l1, l2, l3
      INTEGER, DIMENSION(:), POINTER                     :: npts
      REAL(KIND=dp), POINTER                             :: ptr_pwout
      INTERFACE
         SUBROUTINE pw_gpu_fc_c(zin, dout, npts) BIND(C, name="pw_gpu_fc")
            IMPORT
            TYPE(C_PTR), INTENT(IN), VALUE               :: zin
            TYPE(C_PTR), VALUE                           :: dout
            INTEGER(KIND=C_INT), DIMENSION(*), INTENT(IN):: npts
         END SUBROUTINE pw_gpu_fc_c
      END INTERFACE

      CALL timeset(routineN, handle)

      npts => pw2%pw_grid%npts_local
      l1 = LBOUND(pw2%array, 1)
      l2 = LBOUND(pw2%array, 2)
      l3 = LBOUND(pw2%array, 3)

      ! pointers to data arrays
      ptr_pwin => pwbuf(1, 1)
      ptr_pwout => pw2%array(l1, l2, l3)

      ! invoke the combined transformation
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      CALL pw_gpu_fc_c(c_loc(ptr_pwin), c_loc(ptr_pwout), npts)
#else
      CPABORT("Compiled without pw offloading")
#endif

      CALL timestop(handle)
   END SUBROUTINE pw_gpu_fc

! **************************************************************************************************
!> \brief perform a parallel 1D-FFT on the gpu
!> \param pwbuf1 ...
!> \param pwbuf2 ...
!> \param dir ...
!> \param n ...
!> \param m ...
!> \author Andreas Gloess
! **************************************************************************************************
   SUBROUTINE pw_gpu_f(pwbuf1, pwbuf2, dir, n, m)
      COMPLEX(KIND=dp), DIMENSION(:, :), INTENT(IN), &
         TARGET                                          :: pwbuf1
      COMPLEX(KIND=dp), DIMENSION(:, :), INTENT(INOUT), &
         TARGET                                          :: pwbuf2
      INTEGER, INTENT(IN)                                :: dir, n, m

      CHARACTER(len=*), PARAMETER                        :: routineN = 'pw_gpu_f'

      COMPLEX(KIND=dp), POINTER                          :: ptr_pwin, ptr_pwout
      INTEGER                                            :: handle
      INTERFACE
         SUBROUTINE pw_gpu_f_c(zin, zout, dir, n, m) BIND(C, name="pw_gpu_f")
            IMPORT
            TYPE(C_PTR), INTENT(IN), VALUE               :: zin
            TYPE(C_PTR), VALUE                           :: zout
            INTEGER(KIND=C_INT), INTENT(IN), VALUE       :: dir, n, m
         END SUBROUTINE pw_gpu_f_c
      END INTERFACE

      CALL timeset(routineN, handle)

      IF (n*m /= 0) THEN
         ! pointers to data arrays
         ptr_pwin => pwbuf1(1, 1)
         ptr_pwout => pwbuf2(1, 1)

         ! invoke the combined transformation
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
         CALL pw_gpu_f_c(c_loc(ptr_pwin), c_loc(ptr_pwout), dir, n, m)
#else
         MARK_USED(dir)
         CPABORT("Compiled without pw offloading")
#endif
      END IF

      CALL timestop(handle)
   END SUBROUTINE pw_gpu_f
! **************************************************************************************************
!> \brief perform a parallel 1D-FFT followed by a gather on the gpu
!> \param pwbuf ...
!> \param pw2 ...
!> \param scale ...
!> \author Andreas Gloess
! **************************************************************************************************
   SUBROUTINE pw_gpu_fg(pwbuf, pw2, scale)
      COMPLEX(KIND=dp), DIMENSION(:, :), INTENT(IN), &
         TARGET                                          :: pwbuf
      TYPE(pw_c1d_gs_type), INTENT(IN)                   :: pw2
      REAL(KIND=dp), INTENT(IN)                          :: scale

      CHARACTER(len=*), PARAMETER                        :: routineN = 'pw_gpu_fg'

      COMPLEX(KIND=dp), POINTER                          :: ptr_pwin, ptr_pwout
      INTEGER                                            :: handle, mg, mmax, ngpts
      INTEGER, DIMENSION(:), POINTER                     :: npts
      INTEGER, POINTER                                   :: ptr_ghatmap
      INTERFACE
         SUBROUTINE pw_gpu_fg_c(zin, zout, ghatmap, npts, mmax, ngpts, scale) BIND(C, name="pw_gpu_fg")
            IMPORT
            TYPE(C_PTR), INTENT(IN), VALUE               :: zin
            TYPE(C_PTR), VALUE                           :: zout
            TYPE(C_PTR), INTENT(IN), VALUE               :: ghatmap
            INTEGER(KIND=C_INT), DIMENSION(*), INTENT(IN):: npts
            INTEGER(KIND=C_INT), INTENT(IN), VALUE       :: mmax, ngpts
            REAL(KIND=C_DOUBLE), INTENT(IN), VALUE       :: scale

         END SUBROUTINE pw_gpu_fg_c
      END INTERFACE

      CALL timeset(routineN, handle)

      ngpts = SIZE(pw2%pw_grid%gsq)
      npts => pw2%pw_grid%npts

      IF ((npts(1) /= 0) .AND. (ngpts /= 0)) THEN
         mg = SIZE(pw2%pw_grid%grays, 2)
         mmax = MAX(mg, 1)

         ! pointers to data arrays
         ptr_pwin => pwbuf(1, 1)
         ptr_pwout => pw2%array(1)

         ! pointer to map array
         ptr_ghatmap => pw2%pw_grid%g_hatmap(1, 1)

         ! invoke the combined transformation
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
         CALL pw_gpu_fg_c(c_loc(ptr_pwin), c_loc(ptr_pwout), c_loc(ptr_ghatmap), npts, mmax, ngpts, scale)
#else
         MARK_USED(scale)
         CPABORT("Compiled without pw offloading")
#endif
      END IF

      CALL timestop(handle)
   END SUBROUTINE pw_gpu_fg

! **************************************************************************************************
!> \brief perform a parallel scatter followed by a 1D-FFT on the gpu
!> \param pw1 ...
!> \param pwbuf ...
!> \param scale ...
!> \author Andreas Gloess
! **************************************************************************************************
   SUBROUTINE pw_gpu_sf(pw1, pwbuf, scale)
      TYPE(pw_c1d_gs_type), INTENT(IN)                   :: pw1
      COMPLEX(KIND=dp), DIMENSION(:, :), INTENT(INOUT), &
         TARGET                                          :: pwbuf
      REAL(KIND=dp), INTENT(IN)                          :: scale

      CHARACTER(len=*), PARAMETER                        :: routineN = 'pw_gpu_sf'

      COMPLEX(KIND=dp), POINTER                          :: ptr_pwin, ptr_pwout
      INTEGER                                            :: handle, mg, mmax, ngpts, nmaps
      INTEGER, DIMENSION(:), POINTER                     :: npts
      INTEGER, POINTER                                   :: ptr_ghatmap
      INTERFACE
         SUBROUTINE pw_gpu_sf_c(zin, zout, ghatmap, npts, mmax, ngpts, nmaps, scale) BIND(C, name="pw_gpu_sf")
            IMPORT
            TYPE(C_PTR), INTENT(IN), VALUE               :: zin
            TYPE(C_PTR), VALUE                           :: zout
            TYPE(C_PTR), INTENT(IN), VALUE               :: ghatmap
            INTEGER(KIND=C_INT), DIMENSION(*), INTENT(IN):: npts
            INTEGER(KIND=C_INT), INTENT(IN), VALUE       :: mmax, ngpts, nmaps
            REAL(KIND=C_DOUBLE), INTENT(IN), VALUE       :: scale

         END SUBROUTINE pw_gpu_sf_c
      END INTERFACE

      CALL timeset(routineN, handle)

      ngpts = SIZE(pw1%pw_grid%gsq)
      npts => pw1%pw_grid%npts

      IF ((npts(1) /= 0) .AND. (ngpts /= 0)) THEN
         mg = SIZE(pw1%pw_grid%grays, 2)
         mmax = MAX(mg, 1)

         ! pointers to data arrays
         ptr_pwin => pw1%array(1)
         ptr_pwout => pwbuf(1, 1)

         ! pointer to map array
         nmaps = SIZE(pw1%pw_grid%g_hatmap, 2)
         ptr_ghatmap => pw1%pw_grid%g_hatmap(1, 1)

         ! invoke the combined transformation
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
         CALL pw_gpu_sf_c(c_loc(ptr_pwin), c_loc(ptr_pwout), c_loc(ptr_ghatmap), npts, mmax, ngpts, nmaps, scale)
#else
         MARK_USED(scale)
         CPABORT("Compiled without pw offloading")
#endif
      END IF

      CALL timestop(handle)
   END SUBROUTINE pw_gpu_sf

! **************************************************************************************************
!> \brief Checks whether the fused (device-resident) XC paths can be used on a grid:
!>        runtime switch CP2K_PW_GPU_FUSED_XC on, non-distributed grid with a valid id.
!> \param pw_grid ...
!> \return ...
! **************************************************************************************************
   FUNCTION pw_gpu_fused_xc_available(pw_grid) RESULT(res)
      TYPE(pw_grid_type), INTENT(IN)                     :: pw_grid
      LOGICAL                                            :: res

      INTERFACE
         FUNCTION pw_gpu_fused_xc_enabled_c() BIND(C, name="pw_gpu_fused_xc_enabled") RESULT(r)
            IMPORT :: C_INT
            INTEGER(KIND=C_INT)                          :: r
         END FUNCTION pw_gpu_fused_xc_enabled_c
      END INTERFACE

      res = .FALSE.
      IF (pw_grid%para%mode /= PW_MODE_LOCAL .OR. pw_grid%id_nr <= 0) RETURN
      IF (.NOT. (ASSOCIATED(pw_grid%g) .AND. ASSOCIATED(pw_grid%g_hatmap))) RETURN
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      res = (pw_gpu_fused_xc_enabled_c() /= 0)
#endif
   END FUNCTION pw_gpu_fused_xc_available

! **************************************************************************************************
!> \brief Full XC uses the same serial, local-grid ownership as the fused FFT paths.
! **************************************************************************************************
   FUNCTION pw_gpu_xc_full_pipeline_available(pw_grid) RESULT(res)
      TYPE(pw_grid_type), INTENT(IN) :: pw_grid
      LOGICAL :: res

      res = pw_gpu_fused_xc_available(pw_grid)
   END FUNCTION pw_gpu_xc_full_pipeline_available

! **************************************************************************************************
!> \brief Try gradient, PBE, divergence, filtering and energy reduction entirely in VRAM.
!>        Only the input densities and final potential/energy cross the host boundary.
!>        A false result leaves both outputs untouched for the original XC path.
! **************************************************************************************************
   LOGICAL FUNCTION pw_gpu_xc_full_pipeline(rho_g, rho_r, vxc, exc, epsilon_rho, &
                                          kappa, beta, mu, scale_ec, scale_ex, &
                                          gradient_cutoff, smooth_range) RESULT(done)
      TYPE(pw_c1d_gs_type), INTENT(IN) :: rho_g
      TYPE(pw_r3d_rs_type), INTENT(IN) :: rho_r
      TYPE(pw_r3d_rs_type), INTENT(INOUT) :: vxc
      REAL(KIND=dp), INTENT(INOUT) :: exc
      REAL(KIND=dp), INTENT(IN) :: epsilon_rho, kappa, beta, mu, scale_ec, scale_ex
      REAL(KIND=dp), INTENT(IN) :: gradient_cutoff, smooth_range

      TYPE(pw_grid_type), POINTER :: grid
      INTEGER :: ngpts, nmaps, d
      INTEGER(KIND=C_SIZE_T) :: nrpts
      REAL(KIND=dp) :: energy_sum, scale
      INTERFACE
         FUNCTION pw_gpu_xc_full_pipeline_c(zin, din, dout, energy, map, g, npts, ng, nm, scale, id, &
                                            epsilon, kappa, beta, mu, sc, sx, drho_cutoff, smooth, filter) &
            BIND(C, name="pw_gpu_xc_full_pipeline_configured_c") RESULT(res)
            IMPORT :: C_PTR, C_INT, C_DOUBLE
            TYPE(C_PTR), VALUE :: zin, din, dout, map, g
            REAL(KIND=C_DOUBLE), INTENT(OUT) :: energy
            INTEGER(KIND=C_INT), INTENT(IN) :: npts(3)
            INTEGER(KIND=C_INT), VALUE :: ng, nm, id, filter
            REAL(KIND=C_DOUBLE), VALUE :: scale, epsilon, kappa, beta, mu, sc, sx, drho_cutoff, smooth
            INTEGER(KIND=C_INT) :: res
         END FUNCTION pw_gpu_xc_full_pipeline_c
      END INTERFACE

      done = .FALSE.
      grid => rho_r%pw_grid
      IF (.NOT. ASSOCIATED(grid)) RETURN
      IF (.NOT. pw_gpu_xc_full_pipeline_available(grid)) RETURN
      IF (.NOT. ASSOCIATED(rho_g%pw_grid, grid) .OR. .NOT. ASSOCIATED(vxc%pw_grid, grid)) RETURN
      IF (.NOT. ASSOCIATED(rho_g%array) .OR. .NOT. ASSOCIATED(rho_r%array) .OR. &
          .NOT. ASSOCIATED(vxc%array)) RETURN
      IF (.NOT. IS_CONTIGUOUS(rho_g%array) .OR. .NOT. IS_CONTIGUOUS(rho_r%array) .OR. &
          .NOT. IS_CONTIGUOUS(vxc%array)) RETURN
      IF (.NOT. IS_CONTIGUOUS(grid%g) .OR. .NOT. IS_CONTIGUOUS(grid%g_hatmap)) RETURN
      nrpts = 1_C_SIZE_T
      DO d = 1, 3
         IF (grid%npts(d) <= 0) RETURN
         IF (nrpts > INT(HUGE(0_C_INT)/2, C_SIZE_T)/INT(grid%npts(d), C_SIZE_T)) RETURN
         nrpts = nrpts*INT(grid%npts(d), C_SIZE_T)
      END DO
      IF (SIZE(rho_r%array, KIND=C_SIZE_T) /= nrpts .OR. SIZE(vxc%array, KIND=C_SIZE_T) /= nrpts) RETURN
      IF (ANY(SHAPE(rho_r%array) /= grid%npts) .OR. ANY(SHAPE(vxc%array) /= grid%npts)) RETURN
      ngpts = SIZE(rho_g%array)
      nmaps = SIZE(grid%g_hatmap, 2)
      IF (ngpts <= 0 .OR. ngpts > nrpts) RETURN
      IF (SIZE(grid%g, 1) /= 3 .OR. SIZE(grid%g, 2) /= ngpts) RETURN
      IF (SIZE(grid%g_hatmap, 1) /= ngpts .OR. (nmaps /= 1 .AND. nmaps /= 2)) RETURN
      scale = 1.0_dp/REAL(nrpts, dp)
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      done = pw_gpu_xc_full_pipeline_c(C_LOC(rho_g%array), C_LOC(rho_r%array), C_LOC(vxc%array), &
                                      energy_sum, C_LOC(grid%g_hatmap), C_LOC(grid%g), INT(grid%npts, C_INT), &
                                      INT(ngpts, C_INT), INT(nmaps, C_INT), scale, INT(grid%id_nr, C_INT), &
                                      epsilon_rho, kappa, beta, mu, scale_ec, scale_ex, gradient_cutoff, &
                                      smooth_range, MERGE(1_C_INT, 0_C_INT, grid%spherical)) /= 0_C_INT
      IF (done) exc = energy_sum*grid%dvol
#endif
   END FUNCTION pw_gpu_xc_full_pipeline

! **************************************************************************************************
!> \brief Fused XC gradient on the GPU: grad(d) = FFT^-1[ i G_d pw_g ], d = 1..3.
!>        pw_g is uploaded once; derivative, scatter and inverse FFT run on the device.
!>        Equivalent (bitwise) to 3 x { pw_zero, pw_copy, pw_derive, pw_transfer }.
!>        All arguments must live on the same, non-distributed pw_grid.
!> \param pw_g ...
!> \param grad ...
! **************************************************************************************************
   SUBROUTINE pw_gpu_gradient_c1d_r3d(pw_g, grad)
      TYPE(pw_c1d_gs_type), INTENT(IN)                   :: pw_g
      TYPE(pw_r3d_rs_type), DIMENSION(3), INTENT(INOUT)  :: grad

      CHARACTER(len=*), PARAMETER :: routineN = 'pw_gpu_gradient_c1d_r3d'

      COMPLEX(KIND=dp), POINTER                          :: ptr_pwin
      INTEGER                                            :: handle, idir, ngpts, nmaps
      INTEGER, DIMENSION(:), POINTER                     :: npts
      INTEGER, POINTER                                   :: ptr_ghatmap
      REAL(KIND=dp), POINTER                             :: ptr_g, ptr_o1, ptr_o2, ptr_o3
      INTERFACE
         SUBROUTINE pw_gpu_sfffc_grad3_c(zin, dout1, dout2, dout3, ghatmap, g, npts, ngpts, nmaps, grid_id) &
            BIND(C, name="pw_gpu_sfffc_grad3")
            IMPORT
            TYPE(C_PTR), INTENT(IN), VALUE               :: zin
            TYPE(C_PTR), VALUE                           :: dout1, dout2, dout3
            TYPE(C_PTR), INTENT(IN), VALUE               :: ghatmap, g
            INTEGER(KIND=C_INT), DIMENSION(*), INTENT(IN):: npts
            INTEGER(KIND=C_INT), INTENT(IN), VALUE       :: ngpts, nmaps, grid_id
         END SUBROUTINE pw_gpu_sfffc_grad3_c
      END INTERFACE

      CALL timeset(routineN, handle)

      npts => pw_g%pw_grid%npts
      ngpts = SIZE(pw_g%pw_grid%gsq)
      DO idir = 1, 3
         CPASSERT(ASSOCIATED(grad(idir)%pw_grid, pw_g%pw_grid))
         CPASSERT(SIZE(grad(idir)%array) == npts(1)*npts(2)*npts(3))
      END DO

      ptr_pwin => pw_g%array(1)
      ptr_o1 => grad(1)%array(LBOUND(grad(1)%array, 1), LBOUND(grad(1)%array, 2), LBOUND(grad(1)%array, 3))
      ptr_o2 => grad(2)%array(LBOUND(grad(2)%array, 1), LBOUND(grad(2)%array, 2), LBOUND(grad(2)%array, 3))
      ptr_o3 => grad(3)%array(LBOUND(grad(3)%array, 1), LBOUND(grad(3)%array, 2), LBOUND(grad(3)%array, 3))
      nmaps = SIZE(pw_g%pw_grid%g_hatmap, 2)
      ptr_ghatmap => pw_g%pw_grid%g_hatmap(1, 1)
      ptr_g => pw_g%pw_grid%g(1, 1)

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      CALL pw_gpu_sfffc_grad3_c(c_loc(ptr_pwin), c_loc(ptr_o1), c_loc(ptr_o2), c_loc(ptr_o3), &
                                c_loc(ptr_ghatmap), c_loc(ptr_g), npts, ngpts, nmaps, pw_g%pw_grid%id_nr)
#else
      CPABORT("Compiled without pw offloading")
#endif

      CALL timestop(handle)
   END SUBROUTINE pw_gpu_gradient_c1d_r3d

! **************************************************************************************************
!> \brief Fused XC divergence on the GPU:
!>        pw_out = FFT^-1[ sum_d i G_d FFT[pw_in(d)] ]  (G-space sum stays on the device).
!>        Equivalent (bitwise) to the pw branch of xc_pw_divergence up to (excluding) the
!>        final pw_axpy into vxc_r. pw_out may be pw_in(1).
!>        All arguments must live on the same, non-distributed pw_grid.
!> \param pw_in ...
!> \param pw_out ...
! **************************************************************************************************
   SUBROUTINE pw_gpu_divergence_r3d(pw_in, pw_out)
      TYPE(pw_r3d_rs_type), DIMENSION(3), INTENT(IN)     :: pw_in
      TYPE(pw_r3d_rs_type), INTENT(IN)                   :: pw_out

      CHARACTER(len=*), PARAMETER :: routineN = 'pw_gpu_divergence_r3d'

      INTEGER                                            :: handle, idir, ngpts, nmaps
      INTEGER, DIMENSION(:), POINTER                     :: npts
      INTEGER, POINTER                                   :: ptr_ghatmap
      REAL(KIND=dp)                                      :: scale
      REAL(KIND=dp), POINTER                             :: ptr_g, ptr_i1, ptr_i2, ptr_i3, ptr_out
      INTERFACE
         SUBROUTINE pw_gpu_cfffg_div3_c(din1, din2, din3, dout, ghatmap, g, npts, ngpts, nmaps, scale, grid_id) &
            BIND(C, name="pw_gpu_cfffg_div3")
            IMPORT
            TYPE(C_PTR), INTENT(IN), VALUE               :: din1, din2, din3
            TYPE(C_PTR), VALUE                           :: dout
            TYPE(C_PTR), INTENT(IN), VALUE               :: ghatmap, g
            INTEGER(KIND=C_INT), DIMENSION(*), INTENT(IN):: npts
            INTEGER(KIND=C_INT), INTENT(IN), VALUE       :: ngpts, nmaps
            REAL(KIND=C_DOUBLE), INTENT(IN), VALUE       :: scale
            INTEGER(KIND=C_INT), INTENT(IN), VALUE       :: grid_id
         END SUBROUTINE pw_gpu_cfffg_div3_c
      END INTERFACE

      CALL timeset(routineN, handle)

      npts => pw_out%pw_grid%npts
      ngpts = SIZE(pw_out%pw_grid%gsq)
      ! same scale as pw_gpu_r3dc1d_3d
      scale = 1.0_dp/REAL(pw_out%pw_grid%ngpts, KIND=dp)
      CPASSERT(SIZE(pw_out%array) == npts(1)*npts(2)*npts(3))
      DO idir = 1, 3
         CPASSERT(ASSOCIATED(pw_in(idir)%pw_grid, pw_out%pw_grid))
         CPASSERT(SIZE(pw_in(idir)%array) == npts(1)*npts(2)*npts(3))
      END DO

      ptr_i1 => pw_in(1)%array(LBOUND(pw_in(1)%array, 1), LBOUND(pw_in(1)%array, 2), LBOUND(pw_in(1)%array, 3))
      ptr_i2 => pw_in(2)%array(LBOUND(pw_in(2)%array, 1), LBOUND(pw_in(2)%array, 2), LBOUND(pw_in(2)%array, 3))
      ptr_i3 => pw_in(3)%array(LBOUND(pw_in(3)%array, 1), LBOUND(pw_in(3)%array, 2), LBOUND(pw_in(3)%array, 3))
      ptr_out => pw_out%array(LBOUND(pw_out%array, 1), LBOUND(pw_out%array, 2), LBOUND(pw_out%array, 3))
      nmaps = SIZE(pw_out%pw_grid%g_hatmap, 2)
      ptr_ghatmap => pw_out%pw_grid%g_hatmap(1, 1)
      ptr_g => pw_out%pw_grid%g(1, 1)

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      CALL pw_gpu_cfffg_div3_c(c_loc(ptr_i1), c_loc(ptr_i2), c_loc(ptr_i3), c_loc(ptr_out), &
                               c_loc(ptr_ghatmap), c_loc(ptr_g), npts, ngpts, nmaps, scale, &
                               pw_out%pw_grid%id_nr)
#else
      MARK_USED(scale)
      CPABORT("Compiled without pw offloading")
#endif

      CALL timestop(handle)
   END SUBROUTINE pw_gpu_divergence_r3d

! **************************************************************************************************
!> \brief Fused spherical cutoff filter of a real space grid on the GPU (in place):
!>        pw = FFT^-1[ P_sphere FFT[pw] ]. Equivalent (bitwise) to
!>        pw_transfer(pw, pw_g) + pw_transfer(pw_g, pw) on the same non-distributed grid.
!> \param pw ...
! **************************************************************************************************
   SUBROUTINE pw_gpu_filter_r3d(pw)
      TYPE(pw_r3d_rs_type), INTENT(IN)                   :: pw

      CHARACTER(len=*), PARAMETER                        :: routineN = 'pw_gpu_filter_r3d'

      INTEGER                                            :: handle, ngpts, nmaps
      INTEGER, DIMENSION(:), POINTER                     :: npts
      INTEGER, POINTER                                   :: ptr_ghatmap
      REAL(KIND=dp)                                      :: scale
      REAL(KIND=dp), POINTER                             :: ptr_data
      INTERFACE
         SUBROUTINE pw_gpu_filter_r3d_c(data, ghatmap, npts, ngpts, nmaps, scale, grid_id) &
            BIND(C, name="pw_gpu_filter_r3d")
            IMPORT
            TYPE(C_PTR), VALUE                           :: data
            TYPE(C_PTR), INTENT(IN), VALUE               :: ghatmap
            INTEGER(KIND=C_INT), DIMENSION(*), INTENT(IN):: npts
            INTEGER(KIND=C_INT), INTENT(IN), VALUE       :: ngpts, nmaps
            REAL(KIND=C_DOUBLE), INTENT(IN), VALUE       :: scale
            INTEGER(KIND=C_INT), INTENT(IN), VALUE       :: grid_id
         END SUBROUTINE pw_gpu_filter_r3d_c
      END INTERFACE

      CALL timeset(routineN, handle)

      npts => pw%pw_grid%npts
      ngpts = SIZE(pw%pw_grid%gsq)
      scale = 1.0_dp/REAL(pw%pw_grid%ngpts, KIND=dp)
      CPASSERT(SIZE(pw%array) == npts(1)*npts(2)*npts(3))

      ptr_data => pw%array(LBOUND(pw%array, 1), LBOUND(pw%array, 2), LBOUND(pw%array, 3))
      nmaps = SIZE(pw%pw_grid%g_hatmap, 2)
      ptr_ghatmap => pw%pw_grid%g_hatmap(1, 1)

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      CALL pw_gpu_filter_r3d_c(c_loc(ptr_data), c_loc(ptr_ghatmap), npts, ngpts, nmaps, scale, &
                               pw%pw_grid%id_nr)
#else
      MARK_USED(scale)
      CPABORT("Compiled without pw offloading")
#endif

      CALL timestop(handle)
   END SUBROUTINE pw_gpu_filter_r3d

! **************************************************************************************************
!> \brief Device residency of the PW chain density -> Hartree/DDAPC -> XC -> potential.
!>        Every routine returns .FALSE. (with all outputs untouched) when the operation is not
!>        supported or disabled; the caller then executes the original code.
!>        Switches (environment): CP2K_PW_GPU_RES, CP2K_PW_GPU_RES_{DENSITY,HARTREE,XCIN,POT}.
! **************************************************************************************************

! **************************************************************************************************
!> \brief Whether a residency feature (pw_gpu_res_* constants) is enabled at run time.
!> \param what ...
!> \return ...
! **************************************************************************************************
   FUNCTION pw_gpu_res_enabled(what) RESULT(res)
      INTEGER, INTENT(IN)                                :: what
      LOGICAL                                            :: res

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      INTERFACE
         FUNCTION pw_gpu_res_enabled_c(what) BIND(C, name="pw_gpu_res_enabled_c") RESULT(r)
            IMPORT :: C_INT
            INTEGER(KIND=C_INT), VALUE                   :: what
            INTEGER(KIND=C_INT)                          :: r
         END FUNCTION pw_gpu_res_enabled_c
      END INTERFACE

      res = pw_gpu_res_enabled_c(INT(what, C_INT)) /= 0_C_INT
#else
      MARK_USED(what)
      res = .FALSE.
#endif
   END FUNCTION pw_gpu_res_enabled

! **************************************************************************************************
!> \brief Forgets all device copies (they would also be rejected by their fingerprints).
! **************************************************************************************************
   SUBROUTINE pw_gpu_res_invalidate()
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      INTERFACE
         SUBROUTINE pw_gpu_res_invalidate_c() BIND(C, name="pw_gpu_res_invalidate_c")
         END SUBROUTINE pw_gpu_res_invalidate_c
      END INTERFACE

      CALL pw_gpu_res_invalidate_c()
#endif
   END SUBROUTINE pw_gpu_res_invalidate

! **************************************************************************************************
!> \brief Opens (on) or closes the lazy window of an SCF. Closing writes every result that is
!>        still only on the device to its host array.
!> \param on ...
! **************************************************************************************************
   SUBROUTINE pw_gpu_res_set_lazy(on)
      LOGICAL, INTENT(IN)                                :: on

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      INTERFACE
         SUBROUTINE pw_gpu_res_set_lazy_c(on) BIND(C, name="pw_gpu_res_set_lazy_c")
            IMPORT :: C_INT
            INTEGER(KIND=C_INT), VALUE                   :: on
         END SUBROUTINE pw_gpu_res_set_lazy_c
      END INTERFACE

      IF (on) THEN
         CALL pw_gpu_res_set_lazy_c(1_C_INT)
      ELSE
         CALL pw_gpu_res_set_lazy_c(0_C_INT)
      END IF
#else
      MARK_USED(on)
#endif
   END SUBROUTINE pw_gpu_res_set_lazy

! **************************************************************************************************
!> \brief Lets the next density computation leave its result on the device (lazy window only).
!> \param on ...
! **************************************************************************************************
   SUBROUTINE pw_gpu_res_defer_density(on)
      LOGICAL, INTENT(IN)                                :: on

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      INTERFACE
         SUBROUTINE pw_gpu_res_defer_density_c(on) BIND(C, name="pw_gpu_res_defer_density_c")
            IMPORT :: C_INT
            INTEGER(KIND=C_INT), VALUE                   :: on
         END SUBROUTINE pw_gpu_res_defer_density_c
      END INTERFACE

      IF (on) THEN
         CALL pw_gpu_res_defer_density_c(1_C_INT)
      ELSE
         CALL pw_gpu_res_defer_density_c(0_C_INT)
      END IF
#else
      MARK_USED(on)
#endif
   END SUBROUTINE pw_gpu_res_defer_density

! **************************************************************************************************
!> \brief Forgets the device copy of an array that is released without being read again.
!> \param pw ...
! **************************************************************************************************
   SUBROUTINE pw_gpu_res_discard(pw)
      TYPE(pw_r3d_rs_type), INTENT(IN)                   :: pw

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      INTERFACE
         SUBROUTINE pw_gpu_res_discard_c(host) BIND(C, name="pw_gpu_res_discard_c")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE                           :: host
         END SUBROUTINE pw_gpu_res_discard_c
      END INTERFACE

      IF (.NOT. ASSOCIATED(pw%array)) RETURN
      IF (SIZE(pw%array) == 0) RETURN
      CALL pw_gpu_res_discard_c(C_LOC(pw%array))
#else
      MARK_USED(pw)
#endif
   END SUBROUTINE pw_gpu_res_discard

! **************************************************************************************************
!> \brief Materialize a lazy PW array before host access (also accepts an array slice).
! **************************************************************************************************
   SUBROUTINE pw_gpu_res_ensure_host_ptr(host)
      TYPE(C_PTR), INTENT(IN) :: host
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      INTERFACE
         SUBROUTINE pw_gpu_res_ensure_host_c(id, host) BIND(C, name="pw_gpu_res_ensure_host_c")
            IMPORT :: C_INT, C_PTR
            INTEGER(KIND=C_INT), VALUE :: id
            TYPE(C_PTR), VALUE :: host
         END SUBROUTINE pw_gpu_res_ensure_host_c
      END INTERFACE
      CALL pw_gpu_res_ensure_host_c(-1_C_INT, host)
#else
      MARK_USED(host)
#endif
   END SUBROUTINE pw_gpu_res_ensure_host_ptr

! **************************************************************************************************
!> \brief Synchronize a real-space density or potential before a host consumer uses it.
! **************************************************************************************************
   SUBROUTINE pw_gpu_res_ensure_host_r(pw)
      TYPE(pw_r3d_rs_type), INTENT(IN) :: pw
      IF (.NOT. ASSOCIATED(pw%array)) RETURN
      IF (SIZE(pw%array) == 0) RETURN
      CALL pw_gpu_res_ensure_host_ptr(C_LOC(pw%array))
   END SUBROUTINE pw_gpu_res_ensure_host_r

! **************************************************************************************************
!> \brief Synchronize a reciprocal-space density before a host consumer uses it.
! **************************************************************************************************
   SUBROUTINE pw_gpu_res_ensure_host_g(pw)
      TYPE(pw_c1d_gs_type), INTENT(IN) :: pw
      IF (.NOT. ASSOCIATED(pw%array)) RETURN
      IF (SIZE(pw%array) == 0) RETURN
      CALL pw_gpu_res_ensure_host_ptr(C_LOC(pw%array))
   END SUBROUTINE pw_gpu_res_ensure_host_g

! **************************************************************************************************
!> \brief Integrate the resident electron density with the isign=-1 charge convention.
!>        Other arrays and distributed grids retain pw_integrate_function's CPU path.
! **************************************************************************************************
   FUNCTION pw_gpu_res_integrate_r(rho, total_rho) RESULT(done)
      TYPE(pw_r3d_rs_type), INTENT(IN) :: rho
      REAL(KIND=dp), INTENT(INOUT) :: total_rho
      LOGICAL :: done
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      INTERFACE
         FUNCTION pw_gpu_res_matches_c(id, host, n) BIND(C, name="pw_gpu_res_matches_c") RESULT(res)
            IMPORT :: C_INT, C_PTR, C_SIZE_T
            INTEGER(KIND=C_INT), VALUE :: id
            TYPE(C_PTR), VALUE :: host
            INTEGER(KIND=C_SIZE_T), VALUE :: n
            INTEGER(KIND=C_INT) :: res
         END FUNCTION pw_gpu_res_matches_c
         FUNCTION pw_gpu_res_integrate_r_c(total, dvol) BIND(C, name="pw_gpu_res_integrate_r_c") RESULT(res)
            IMPORT :: C_INT, C_DOUBLE
            REAL(KIND=C_DOUBLE), INTENT(INOUT) :: total
            REAL(KIND=C_DOUBLE), VALUE :: dvol
            INTEGER(KIND=C_INT) :: res
         END FUNCTION pw_gpu_res_integrate_r_c
      END INTERFACE
#endif

      done = .FALSE.
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      IF (.NOT. pw_gpu_res_enabled(pw_gpu_res_density_id)) RETURN
      IF (.NOT. res_grid_ok(rho%pw_grid)) RETURN
      IF (.NOT. res_rs_ok(rho, rho%pw_grid)) RETURN
      IF (pw_gpu_res_matches_c(1_C_INT, C_LOC(rho%array), SIZE(rho%array, KIND=C_SIZE_T)) == 0_C_INT) RETURN
      done = pw_gpu_res_integrate_r_c(total_rho, -rho%pw_grid%dvol) /= 0_C_INT
#else
      MARK_USED(rho)
      MARK_USED(total_rho)
#endif
   END FUNCTION pw_gpu_res_integrate_r

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
! **************************************************************************************************
!> \brief A replicated, non-spherical grid with a contiguous one- or two-sided g_hatmap.
!> \param grid ...
!> \return ...
! **************************************************************************************************
   FUNCTION res_grid_ok(grid) RESULT(ok)
      TYPE(pw_grid_type), POINTER                        :: grid
      LOGICAL                                            :: ok

      ok = .FALSE.
      IF (.NOT. ASSOCIATED(grid)) RETURN
      IF (grid%para%mode /= PW_MODE_LOCAL .OR. grid%spherical) RETURN
      IF (.NOT. ASSOCIATED(grid%gsq) .OR. .NOT. ASSOCIATED(grid%g_hatmap)) RETURN
      IF (SIZE(grid%gsq) < 1 .OR. SIZE(grid%g_hatmap, 1) /= SIZE(grid%gsq)) RETURN
      IF (SIZE(grid%g_hatmap, 2) /= 1 .AND. SIZE(grid%g_hatmap, 2) /= 2) RETURN
      IF (grid%ngpts_cut_local /= SIZE(grid%gsq)) RETURN
      IF (ANY(grid%npts <= 0) .OR. grid%id_nr <= 0) RETURN
      ok = .TRUE.
   END FUNCTION res_grid_ok

! **************************************************************************************************
!> \brief A real-space array covering the whole FFT box of grid.
!> \param pw ...
!> \param grid ...
!> \return ...
! **************************************************************************************************
   FUNCTION res_rs_ok(pw, grid) RESULT(ok)
      TYPE(pw_r3d_rs_type), INTENT(IN)                   :: pw
      TYPE(pw_grid_type), POINTER                        :: grid
      LOGICAL                                            :: ok

      ok = .FALSE.
      IF (.NOT. ASSOCIATED(pw%array) .OR. .NOT. ASSOCIATED(pw%pw_grid, grid)) RETURN
      IF (.NOT. IS_CONTIGUOUS(pw%array)) RETURN
      IF (ANY(SHAPE(pw%array) /= grid%npts)) RETURN
      ok = .TRUE.
   END FUNCTION res_rs_ok

! **************************************************************************************************
!> \brief A reciprocal-space array holding exactly the G vectors of grid.
!> \param pw ...
!> \param grid ...
!> \return ...
! **************************************************************************************************
   FUNCTION res_gs_ok(pw, grid) RESULT(ok)
      TYPE(pw_c1d_gs_type), INTENT(IN)                   :: pw
      TYPE(pw_grid_type), POINTER                        :: grid
      LOGICAL                                            :: ok

      ok = .FALSE.
      IF (.NOT. ASSOCIATED(pw%array) .OR. .NOT. ASSOCIATED(pw%pw_grid, grid)) RETURN
      IF (.NOT. IS_CONTIGUOUS(pw%array)) RETURN
      IF (SIZE(pw%array) /= SIZE(grid%gsq)) RETURN
      ok = .TRUE.
   END FUNCTION res_gs_ok
#endif

! **************************************************************************************************
!> \brief density_rs2pw (pw_interp branch) on the GPU, bitwise equal to
!>          pw_zero(rho_gspace); DO l: pw_transfer(mgrid_rspace(l), mgrid_gspace(l));
!>          pw_axpy(mgrid_gspace(l), rho_gspace); END DO; pw_transfer(rho_gspace, rho)
!>        mgrid_gspace only supplies the grids; its arrays are not written.
!> \param mgrid_rspace ...
!> \param mgrid_gspace ...
!> \param rho ...
!> \param rho_gspace ...
!> \param rs_dev_ptrs Optional completed Grid device views with the same FFT layouts.
!> \return ...
! **************************************************************************************************
   FUNCTION pw_gpu_res_density(mgrid_rspace, mgrid_gspace, rho, rho_gspace, rs_dev_ptrs) RESULT(done)
      TYPE(pw_r3d_rs_type), DIMENSION(:), INTENT(IN)     :: mgrid_rspace
      TYPE(pw_c1d_gs_type), DIMENSION(:), INTENT(IN)     :: mgrid_gspace
      TYPE(pw_r3d_rs_type), INTENT(INOUT)                :: rho
      TYPE(pw_c1d_gs_type), INTENT(INOUT)                :: rho_gspace
      TYPE(C_PTR), DIMENSION(:), INTENT(IN), OPTIONAL    :: rs_dev_ptrs
      LOGICAL                                            :: done

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      INTEGER, PARAMETER                                 :: max_levels = 16

      INTEGER                                            :: l, nlevels
      INTEGER(KIND=C_INT), DIMENSION(3*max_levels)       :: npts
      INTEGER(KIND=C_INT), DIMENSION(max_levels)         :: grid_id, ngpts, nmaps
      REAL(KIND=C_DOUBLE), DIMENSION(max_levels)         :: scale
      TYPE(C_PTR), DIMENSION(max_levels)                 :: gidx, maps, r_host, r_dev
      TYPE(pw_grid_type), POINTER                        :: grid, target_grid
      INTERFACE
         FUNCTION pw_gpu_res_density_c(nlevels, r_host, npts, ngpts, nmaps, grid_id, maps, gidx, &
                                       scale, t_npts, t_ngpts, t_nmaps, t_grid_id, t_map, rho_g, rho_r, r_dev) &
            BIND(C, name="pw_gpu_res_density_c") RESULT(res)
            IMPORT :: C_PTR, C_INT, C_DOUBLE
            INTEGER(KIND=C_INT), VALUE                   :: nlevels
            TYPE(C_PTR), DIMENSION(*), INTENT(IN)        :: r_host, r_dev
            INTEGER(KIND=C_INT), DIMENSION(*), INTENT(IN) :: npts, ngpts, nmaps, grid_id
            TYPE(C_PTR), DIMENSION(*), INTENT(IN)        :: maps, gidx
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN) :: scale
            INTEGER(KIND=C_INT), DIMENSION(*), INTENT(IN) :: t_npts
            INTEGER(KIND=C_INT), VALUE                   :: t_ngpts, t_nmaps, t_grid_id
            TYPE(C_PTR), VALUE                           :: t_map, rho_g, rho_r
            INTEGER(KIND=C_INT)                          :: res
         END FUNCTION pw_gpu_res_density_c
      END INTERFACE

      done = .FALSE.
      nlevels = SIZE(mgrid_rspace)
      IF (nlevels < 1 .OR. nlevels > max_levels .OR. SIZE(mgrid_gspace) /= nlevels) RETURN
      r_dev = C_NULL_PTR
      IF (PRESENT(rs_dev_ptrs)) THEN
         IF (SIZE(rs_dev_ptrs) /= nlevels) RETURN
         r_dev(1:nlevels) = rs_dev_ptrs
      END IF
      IF (.NOT. pw_gpu_res_enabled(pw_gpu_res_density_id)) RETURN
      target_grid => rho_gspace%pw_grid
      IF (.NOT. res_grid_ok(target_grid)) RETURN
      IF (.NOT. res_gs_ok(rho_gspace, target_grid) .OR. .NOT. res_rs_ok(rho, target_grid)) RETURN
      DO l = 1, nlevels
         grid => mgrid_gspace(l)%pw_grid
         IF (.NOT. res_grid_ok(grid)) RETURN
         r_host(l) = C_NULL_PTR
         IF (.NOT. C_ASSOCIATED(r_dev(l))) THEN
            IF (.NOT. res_rs_ok(mgrid_rspace(l), grid)) RETURN
            r_host(l) = C_LOC(mgrid_rspace(l)%array)
         END IF
         IF (ASSOCIATED(grid, target_grid)) THEN
            gidx(l) = C_NULL_PTR
         ELSE
            ! pw_axpy branch rho_gspace(gidx(i)) += mgrid_gspace(i): the level references the
            ! target grid and has fewer G vectors.
            IF (grid%reference /= target_grid%id_nr .OR. target_grid%reference == grid%id_nr) RETURN
            IF (SIZE(grid%gsq) >= SIZE(target_grid%gsq)) RETURN
            IF (.NOT. ASSOCIATED(grid%gidx)) RETURN
            IF (SIZE(grid%gidx) < SIZE(grid%gsq) .OR. .NOT. IS_CONTIGUOUS(grid%gidx)) RETURN
            gidx(l) = C_LOC(grid%gidx(1))
         END IF
         npts(3*l - 2:3*l) = INT(grid%npts, C_INT)
         ngpts(l) = INT(SIZE(grid%gsq), C_INT)
         nmaps(l) = INT(SIZE(grid%g_hatmap, 2), C_INT)
         grid_id(l) = INT(grid%id_nr, C_INT)
         maps(l) = C_LOC(grid%g_hatmap(1, 1))
         scale(l) = 1.0_dp/REAL(grid%ngpts, KIND=dp)
      END DO
      done = pw_gpu_res_density_c(INT(nlevels, C_INT), r_host, npts, ngpts, nmaps, grid_id, maps, gidx, &
                                  scale, INT(target_grid%npts, C_INT), INT(SIZE(target_grid%gsq), C_INT), &
                                  INT(SIZE(target_grid%g_hatmap, 2), C_INT), INT(target_grid%id_nr, C_INT), &
                                  C_LOC(target_grid%g_hatmap(1, 1)), C_LOC(rho_gspace%array), &
                                  C_LOC(rho%array), r_dev) /= 0_C_INT
#else
      MARK_USED(mgrid_rspace)
      MARK_USED(mgrid_gspace)
      MARK_USED(rho)
      MARK_USED(rho_gspace)
      MARK_USED(rs_dev_ptrs)
      done = .FALSE.
#endif
   END FUNCTION pw_gpu_res_density

! **************************************************************************************************
!> \brief Consume completed, layout-compatible Grid device views without host copies.
!>        Optional real grids supply any levels that still require a host transfer.
   FUNCTION pw_gpu_res_density_grid(rs_dev_ptrs, mgrid_gspace, rho, rho_gspace, mgrid_rspace) RESULT(done)
      TYPE(C_PTR), DIMENSION(:), INTENT(IN) :: rs_dev_ptrs
      TYPE(pw_c1d_gs_type), DIMENSION(:), INTENT(IN) :: mgrid_gspace
      TYPE(pw_r3d_rs_type), INTENT(INOUT) :: rho
      TYPE(pw_c1d_gs_type), INTENT(INOUT) :: rho_gspace
      TYPE(pw_r3d_rs_type), DIMENSION(:), INTENT(IN), OPTIONAL :: mgrid_rspace
      TYPE(pw_r3d_rs_type) :: unused(SIZE(mgrid_gspace))
      LOGICAL :: done

      IF (PRESENT(mgrid_rspace)) THEN
         done = pw_gpu_res_density(mgrid_rspace, mgrid_gspace, rho, rho_gspace, rs_dev_ptrs)
      ELSE
         done = pw_gpu_res_density(unused, mgrid_gspace, rho, rho_gspace, rs_dev_ptrs)
      END IF
   END FUNCTION pw_gpu_res_density_grid

! **************************************************************************************************
!> \brief potential_pw2rs (pw_interp branch) on the GPU, bitwise equal to
!>          pw_transfer(v_rspace, G); DO l /= aux: pw_copy(G, G_l); pw_transfer(G_l, mgrid_rspace(l));
!>          mgrid_rspace(l) = dvol_l/dvol_aux*mgrid_rspace(l); pw_copy(v_rspace, mgrid_rspace(aux))
!> \param v_rspace ...
!> \param mgrid_rspace ...
!> \param aux the auxiliary (finest) level
!> \param rs_dev Optional device memory of the real-space multigrids, one entry per level
!>        (C_NULL_PTR: write mgrid_rspace). Levels with an entry are written there instead.
!> \return ...
! **************************************************************************************************
   FUNCTION pw_gpu_res_potential(v_rspace, mgrid_rspace, aux, rs_dev) RESULT(done)
      TYPE(pw_r3d_rs_type), INTENT(IN)                   :: v_rspace
      TYPE(pw_r3d_rs_type), DIMENSION(:), INTENT(INOUT)  :: mgrid_rspace
      INTEGER, INTENT(IN)                                :: aux
      TYPE(C_PTR), DIMENSION(:), INTENT(IN), OPTIONAL    :: rs_dev
      LOGICAL                                            :: done

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      INTEGER, PARAMETER                                 :: max_levels = 16

      INTEGER                                            :: l, nlevels
      INTEGER(KIND=C_INT), DIMENSION(3*max_levels)       :: npts
      INTEGER(KIND=C_INT), DIMENSION(max_levels)         :: grid_id, ngpts, nmaps
      REAL(KIND=C_DOUBLE), DIMENSION(max_levels)         :: rscale
      TYPE(C_PTR), DIMENSION(max_levels)                 :: gidx, maps, r_dev, r_host
      TYPE(pw_grid_type), POINTER                        :: aux_grid, grid
      INTERFACE
         FUNCTION pw_gpu_res_potential_c(v_host, nlevels, r_host, npts, ngpts, nmaps, grid_id, maps, &
                                         gidx, rscale, aux, fwd_scale, r_dev) &
            BIND(C, name="pw_gpu_res_potential_c") RESULT(res)
            IMPORT :: C_PTR, C_INT, C_DOUBLE
            TYPE(C_PTR), VALUE                           :: v_host
            INTEGER(KIND=C_INT), VALUE                   :: nlevels
            TYPE(C_PTR), DIMENSION(*), INTENT(IN)        :: r_host
            INTEGER(KIND=C_INT), DIMENSION(*), INTENT(IN) :: npts, ngpts, nmaps, grid_id
            TYPE(C_PTR), DIMENSION(*), INTENT(IN)        :: maps, gidx
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN) :: rscale
            INTEGER(KIND=C_INT), VALUE                   :: aux
            REAL(KIND=C_DOUBLE), VALUE                   :: fwd_scale
            TYPE(C_PTR), DIMENSION(*), INTENT(IN)        :: r_dev
            INTEGER(KIND=C_INT)                          :: res
         END FUNCTION pw_gpu_res_potential_c
      END INTERFACE

      done = .FALSE.
      nlevels = SIZE(mgrid_rspace)
      IF (nlevels < 1 .OR. nlevels > max_levels .OR. aux < 1 .OR. aux > nlevels) RETURN
      r_dev = C_NULL_PTR
      IF (PRESENT(rs_dev)) THEN
         IF (SIZE(rs_dev) /= nlevels) RETURN
         r_dev(1:nlevels) = rs_dev
      END IF
      IF (.NOT. pw_gpu_res_enabled(pw_gpu_res_pot_id)) RETURN
      aux_grid => mgrid_rspace(aux)%pw_grid
      IF (.NOT. res_grid_ok(aux_grid)) RETURN
      IF (.NOT. res_rs_ok(v_rspace, aux_grid)) RETURN
      DO l = 1, nlevels
         grid => mgrid_rspace(l)%pw_grid
         IF (.NOT. res_grid_ok(grid)) RETURN
         IF (.NOT. res_rs_ok(mgrid_rspace(l), grid)) RETURN
         gidx(l) = C_NULL_PTR
         rscale(l) = 1.0_dp
         IF (l /= aux) THEN
            ! pw_copy branch G_l(i) = G(gidx_l(i)): the level references the auxiliary grid.
            IF (ASSOCIATED(grid, aux_grid)) RETURN
            IF (grid%reference /= aux_grid%id_nr) RETURN
            IF (SIZE(grid%gsq) > SIZE(aux_grid%gsq)) RETURN
            IF (.NOT. ASSOCIATED(grid%gidx)) RETURN
            IF (SIZE(grid%gidx) < SIZE(grid%gsq) .OR. .NOT. IS_CONTIGUOUS(grid%gidx)) RETURN
            gidx(l) = C_LOC(grid%gidx(1))
            rscale(l) = mgrid_rspace(l)%pw_grid%dvol/mgrid_rspace(aux)%pw_grid%dvol
         END IF
         npts(3*l - 2:3*l) = INT(grid%npts, C_INT)
         ngpts(l) = INT(SIZE(grid%gsq), C_INT)
         nmaps(l) = INT(SIZE(grid%g_hatmap, 2), C_INT)
         grid_id(l) = INT(grid%id_nr, C_INT)
         maps(l) = C_LOC(grid%g_hatmap(1, 1))
         r_host(l) = C_LOC(mgrid_rspace(l)%array)
      END DO
      done = pw_gpu_res_potential_c(C_LOC(v_rspace%array), INT(nlevels, C_INT), r_host, npts, ngpts, &
                                    nmaps, grid_id, maps, gidx, rscale, INT(aux - 1, C_INT), &
                                    1.0_dp/REAL(v_rspace%pw_grid%ngpts, KIND=dp), r_dev) /= 0_C_INT
#else
      MARK_USED(v_rspace)
      MARK_USED(mgrid_rspace)
      MARK_USED(aux)
      MARK_USED(rs_dev)
      done = .FALSE.
#endif
   END FUNCTION pw_gpu_res_potential

! **************************************************************************************************
!> \brief Hartree/DDAPC chain, step 1: rho_tot = rho_core + rho_g and v_H = rho_tot*influence on
!>        the device (as calc_rho_tot_gspace, pw_poisson_solve). Only rho_tot(1:nprefix) is written
!>        to the host, for the DDAPC charge fit and the G=0 charge.
!> \param rho_core ...
!> \param rho_g ...
!> \param influence ...
!> \param rho_tot ...
!> \param nprefix ...
!> \return ...
! **************************************************************************************************
   FUNCTION pw_gpu_res_hartree_begin(rho_core, rho_g, influence, rho_tot, nprefix) RESULT(done)
      TYPE(pw_c1d_gs_type), INTENT(IN)                   :: rho_core, rho_g, influence
      TYPE(pw_c1d_gs_type), INTENT(INOUT)                :: rho_tot
      INTEGER, INTENT(IN)                                :: nprefix
      LOGICAL                                            :: done

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      TYPE(pw_grid_type), POINTER                        :: grid
      INTERFACE
         FUNCTION pw_gpu_res_hartree_begin_c(core, rho_g, infl, rho_tot, ngpts, nprefix, npts) &
            BIND(C, name="pw_gpu_res_hartree_begin_c") RESULT(res)
            IMPORT :: C_PTR, C_INT, C_SIZE_T
            TYPE(C_PTR), VALUE                           :: core, rho_g, infl, rho_tot
            INTEGER(KIND=C_SIZE_T), VALUE                :: ngpts, nprefix
            INTEGER(KIND=C_INT), DIMENSION(*), INTENT(IN) :: npts
            INTEGER(KIND=C_INT)                          :: res
         END FUNCTION pw_gpu_res_hartree_begin_c
      END INTERFACE

      done = .FALSE.
      IF (.NOT. pw_gpu_res_enabled(pw_gpu_res_hartree_id)) RETURN
      grid => rho_tot%pw_grid
      IF (.NOT. res_grid_ok(grid)) RETURN
      IF (.NOT. res_gs_ok(rho_tot, grid) .OR. .NOT. res_gs_ok(rho_core, grid)) RETURN
      IF (.NOT. res_gs_ok(rho_g, grid) .OR. .NOT. res_gs_ok(influence, grid)) RETURN
      IF (nprefix < 1 .OR. nprefix > SIZE(grid%gsq)) RETURN
      done = pw_gpu_res_hartree_begin_c(C_LOC(rho_core%array), C_LOC(rho_g%array), C_LOC(influence%array), &
                                        C_LOC(rho_tot%array), INT(SIZE(grid%gsq), C_SIZE_T), &
                                        INT(nprefix, C_SIZE_T), INT(grid%npts, C_INT)) /= 0_C_INT
#else
      MARK_USED(rho_core)
      MARK_USED(rho_g)
      MARK_USED(influence)
      MARK_USED(rho_tot)
      MARK_USED(nprefix)
      done = .FALSE.
#endif
   END FUNCTION pw_gpu_res_hartree_begin

! **************************************************************************************************
!> \brief Hartree/DDAPC chain, step 2 (modify_hartree_pot on the device): v_H(1) += g0_delta and
!>        the DDAPC correction for first_gne0 <= ig, |G|^2 <= gcut2 (same kernel as the host-pointer
!>        modify_hartree_pot_gpu).
!> \param g0_delta ...
!> \param grid ...
!> \param positions ...
!> \param radii ...
!> \param cv ...
!> \param gcut2 ...
!> \param vol ...
!> \return ...
! **************************************************************************************************
   FUNCTION pw_gpu_res_hartree_modify(g0_delta, grid, positions, radii, cv, gcut2, vol) RESULT(done)
      REAL(KIND=dp), INTENT(IN)                          :: g0_delta
      TYPE(pw_grid_type), INTENT(IN)                     :: grid
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN), TARGET :: positions
      REAL(KIND=dp), DIMENSION(:), INTENT(IN), TARGET    :: radii, cv
      REAL(KIND=dp), INTENT(IN)                          :: gcut2, vol
      LOGICAL                                            :: done

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      INTEGER                                            :: first, n
      REAL(KIND=dp), DIMENSION(1), TARGET                :: dummy
      TYPE(C_PTR)                                        :: p_cv, p_g, p_gsq, p_r, p_radii
      INTERFACE
         FUNCTION pw_gpu_res_hartree_modify_c(g0_delta, have_g0, g, gsq, r, radii, cv, first, n, natoms, &
                                              nradii, gcut2, vol) &
            BIND(C, name="pw_gpu_res_hartree_modify_c") RESULT(res)
            IMPORT :: C_PTR, C_INT, C_SIZE_T, C_DOUBLE
            REAL(KIND=C_DOUBLE), VALUE                   :: g0_delta
            INTEGER(KIND=C_INT), VALUE                   :: have_g0
            TYPE(C_PTR), VALUE                           :: g, gsq, r, radii, cv
            INTEGER(KIND=C_SIZE_T), VALUE                :: first, n, natoms, nradii
            REAL(KIND=C_DOUBLE), VALUE                   :: gcut2, vol
            INTEGER(KIND=C_INT)                          :: res
         END FUNCTION pw_gpu_res_hartree_modify_c
      END INTERFACE

      done = .FALSE.
      IF (.NOT. ASSOCIATED(grid%g) .OR. .NOT. ASSOCIATED(grid%gsq)) RETURN
      IF (SIZE(positions, 1) /= 3 .OR. SIZE(cv) /= SIZE(positions, 2)*SIZE(radii)) RETURN
      IF (.NOT. IS_CONTIGUOUS(positions) .OR. .NOT. IS_CONTIGUOUS(radii) .OR. .NOT. IS_CONTIGUOUS(cv)) RETURN
      first = grid%first_gne0
      n = MAX(0, MIN(grid%ngpts_cut_local, SIZE(grid%gsq)) - first + 1)
      dummy(1) = 0.0_dp
      p_g = C_LOC(dummy)
      p_gsq = C_LOC(dummy)
      IF (n > 0) THEN
         p_g = C_LOC(grid%g(1, first))
         p_gsq = C_LOC(grid%gsq(first))
      END IF
      p_r = C_LOC(dummy)
      p_radii = C_LOC(dummy)
      p_cv = C_LOC(dummy)
      IF (SIZE(positions) > 0) p_r = C_LOC(positions(1, 1))
      IF (SIZE(radii) > 0) p_radii = C_LOC(radii(1))
      IF (SIZE(cv) > 0) p_cv = C_LOC(cv(1))
      done = pw_gpu_res_hartree_modify_c(g0_delta, MERGE(1_C_INT, 0_C_INT, grid%have_g0), p_g, p_gsq, &
                                         p_r, p_radii, p_cv, INT(MAX(first - 1, 0), C_SIZE_T), &
                                         INT(n, C_SIZE_T), INT(SIZE(positions, 2), C_SIZE_T), &
                                         INT(SIZE(radii), C_SIZE_T), gcut2, vol) /= 0_C_INT
#else
      MARK_USED(g0_delta)
      MARK_USED(grid)
      MARK_USED(positions)
      MARK_USED(radii)
      MARK_USED(cv)
      MARK_USED(gcut2)
      MARK_USED(vol)
      done = .FALSE.
#endif
   END FUNCTION pw_gpu_res_hartree_modify

! **************************************************************************************************
!> \brief Hartree/DDAPC chain, step 3: dot = Re SUM(CONJG(rho_tot)*v_H) (deterministic, compensated;
!>        pw_integral_ab multiplies by the volume).
!> \param dot ...
!> \return ...
! **************************************************************************************************
   FUNCTION pw_gpu_res_hartree_energy(dot) RESULT(done)
      REAL(KIND=dp), INTENT(OUT)                         :: dot
      LOGICAL                                            :: done

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      INTERFACE
         FUNCTION pw_gpu_res_hartree_energy_c(dot) BIND(C, name="pw_gpu_res_hartree_energy_c") RESULT(res)
            IMPORT :: C_INT, C_DOUBLE
            REAL(KIND=C_DOUBLE), INTENT(OUT)             :: dot
            INTEGER(KIND=C_INT)                          :: res
         END FUNCTION pw_gpu_res_hartree_energy_c
      END INTERFACE

      dot = 0.0_dp
      done = pw_gpu_res_hartree_energy_c(dot) /= 0_C_INT
#else
      dot = 0.0_dp
      done = .FALSE.
#endif
   END FUNCTION pw_gpu_res_hartree_energy

! **************************************************************************************************
!> \brief Hartree/DDAPC chain, step 4: v_hartree_rspace = dvol*FFT^-1[v_H] (pw_transfer, pw_scale);
!>        the result stays resident for the potential sum.
!> \param v_hartree_rspace ...
!> \param gs_grid the grid of v_H
!> \return ...
! **************************************************************************************************
   FUNCTION pw_gpu_res_hartree_finish(v_hartree_rspace, gs_grid) RESULT(done)
      TYPE(pw_r3d_rs_type), INTENT(INOUT)                :: v_hartree_rspace
      TYPE(pw_grid_type), POINTER                        :: gs_grid
      LOGICAL                                            :: done

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      INTERFACE
         FUNCTION pw_gpu_res_hartree_finish_c(v_r, npts, ngpts, nmaps, grid_id, map, dvol) &
            BIND(C, name="pw_gpu_res_hartree_finish_c") RESULT(res)
            IMPORT :: C_PTR, C_INT, C_DOUBLE
            TYPE(C_PTR), VALUE                           :: v_r
            INTEGER(KIND=C_INT), DIMENSION(*), INTENT(IN) :: npts
            INTEGER(KIND=C_INT), VALUE                   :: ngpts, nmaps, grid_id
            TYPE(C_PTR), VALUE                           :: map
            REAL(KIND=C_DOUBLE), VALUE                   :: dvol
            INTEGER(KIND=C_INT)                          :: res
         END FUNCTION pw_gpu_res_hartree_finish_c
      END INTERFACE

      done = .FALSE.
      IF (.NOT. res_grid_ok(gs_grid)) RETURN
      IF (.NOT. res_rs_ok(v_hartree_rspace, gs_grid)) RETURN
      done = pw_gpu_res_hartree_finish_c(C_LOC(v_hartree_rspace%array), INT(gs_grid%npts, C_INT), &
                                         INT(SIZE(gs_grid%gsq), C_INT), INT(SIZE(gs_grid%g_hatmap, 2), C_INT), &
                                         INT(gs_grid%id_nr, C_INT), C_LOC(gs_grid%g_hatmap(1, 1)), &
                                         v_hartree_rspace%pw_grid%dvol) /= 0_C_INT
#else
      MARK_USED(v_hartree_rspace)
      MARK_USED(gs_grid)
      done = .FALSE.
#endif
   END FUNCTION pw_gpu_res_hartree_finish

! **************************************************************************************************
!> \brief Abandons a started Hartree/DDAPC chain.
! **************************************************************************************************
   SUBROUTINE pw_gpu_res_hartree_cancel()
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      INTERFACE
         SUBROUTINE pw_gpu_res_hartree_cancel_c() BIND(C, name="pw_gpu_res_hartree_cancel_c")
         END SUBROUTINE pw_gpu_res_hartree_cancel_c
      END INTERFACE

      CALL pw_gpu_res_hartree_cancel_c()
#endif
   END SUBROUTINE pw_gpu_res_hartree_cancel

! **************************************************************************************************
!> \brief To be called right before pw_axpy(v_h, vxc, 1.0_dp, dvol): forms the same sum in the
!>        resident Vxc grid. Returns .FALSE. if Vxc is not resident.
!> \param v_h ...
!> \param vxc ...
!> \param dvol ...
!> \return ...
! **************************************************************************************************
   FUNCTION pw_gpu_res_vtot_prepare(v_h, vxc, dvol) RESULT(done)
      TYPE(pw_r3d_rs_type), INTENT(IN)                   :: v_h, vxc
      REAL(KIND=dp), INTENT(IN)                          :: dvol
      LOGICAL                                            :: done

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      INTERFACE
         FUNCTION pw_gpu_res_vtot_prepare_c(vh, vxc, n, dvol) BIND(C, name="pw_gpu_res_vtot_prepare_c") &
            RESULT(res)
            IMPORT :: C_PTR, C_INT, C_SIZE_T, C_DOUBLE
            TYPE(C_PTR), VALUE                           :: vh, vxc
            INTEGER(KIND=C_SIZE_T), VALUE                :: n
            REAL(KIND=C_DOUBLE), VALUE                   :: dvol
            INTEGER(KIND=C_INT)                          :: res
         END FUNCTION pw_gpu_res_vtot_prepare_c
      END INTERFACE

      done = .FALSE.
      IF (.NOT. pw_gpu_res_enabled(pw_gpu_res_pot_id)) RETURN
      IF (.NOT. ASSOCIATED(v_h%array) .OR. .NOT. ASSOCIATED(vxc%array)) RETURN
      IF (.NOT. ASSOCIATED(v_h%pw_grid, vxc%pw_grid)) RETURN
      IF (.NOT. IS_CONTIGUOUS(v_h%array) .OR. .NOT. IS_CONTIGUOUS(vxc%array)) RETURN
      IF (ANY(SHAPE(v_h%array) /= SHAPE(vxc%array))) RETURN
      done = pw_gpu_res_vtot_prepare_c(C_LOC(v_h%array), C_LOC(vxc%array), SIZE(vxc%array, KIND=C_SIZE_T), &
                                       dvol) /= 0_C_INT
#else
      MARK_USED(v_h)
      MARK_USED(vxc)
      MARK_USED(dvol)
      done = .FALSE.
#endif
   END FUNCTION pw_gpu_res_vtot_prepare

! **************************************************************************************************
!> \brief Accept the prepared device sum and defer host synchronization until first use.
! **************************************************************************************************
   FUNCTION pw_gpu_res_vtot_skip_host() RESULT(done)
      LOGICAL :: done
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      INTERFACE
         FUNCTION pw_gpu_res_vtot_skip_host_c() BIND(C, name="pw_gpu_res_vtot_skip_host_c") RESULT(res)
            IMPORT :: C_INT
            INTEGER(KIND=C_INT) :: res
         END FUNCTION pw_gpu_res_vtot_skip_host_c
      END INTERFACE
      done = pw_gpu_res_vtot_skip_host_c() /= 0_C_INT
#else
      done = .FALSE.
#endif
   END FUNCTION pw_gpu_res_vtot_skip_host

! **************************************************************************************************
!> \brief To be called right after that pw_axpy: keeps the device sum if it equals the host result.
!> \param vtot ...
!> \return ...
! **************************************************************************************************
   FUNCTION pw_gpu_res_vtot_commit(vtot) RESULT(done)
      TYPE(pw_r3d_rs_type), INTENT(IN)                   :: vtot
      LOGICAL                                            :: done

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      INTERFACE
         FUNCTION pw_gpu_res_vtot_commit_c(vtot, n) BIND(C, name="pw_gpu_res_vtot_commit_c") RESULT(res)
            IMPORT :: C_PTR, C_INT, C_SIZE_T
            TYPE(C_PTR), VALUE                           :: vtot
            INTEGER(KIND=C_SIZE_T), VALUE                :: n
            INTEGER(KIND=C_INT)                          :: res
         END FUNCTION pw_gpu_res_vtot_commit_c
      END INTERFACE

      done = .FALSE.
      IF (.NOT. ASSOCIATED(vtot%array)) RETURN
      IF (.NOT. IS_CONTIGUOUS(vtot%array)) RETURN
      done = pw_gpu_res_vtot_commit_c(C_LOC(vtot%array), SIZE(vtot%array, KIND=C_SIZE_T)) /= 0_C_INT
#else
      MARK_USED(vtot)
      done = .FALSE.
#endif
   END FUNCTION pw_gpu_res_vtot_commit

END MODULE pw_gpu

