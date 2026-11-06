!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief This module defines the grid data type and some basic operations on it
!> \note
!>      pw_grid_create : set the defaults
!>      pw_grid_release : release all memory connected to type
!>      pw_grid_setup  : main routine to set up a grid
!>           input: cell (the box for the grid)
!>                  pw_grid (the grid; pw_grid%grid_span has to be set)
!>                  cutoff (optional, if not given pw_grid%bounds has to be set)
!>                  pe_group (optional, if not given we have a local grid)
!>
!>                  if no cutoff or a negative cutoff is given, all g-vectors
!>                  in the box are included (no spherical cutoff)
!>
!>                  for a distributed setup the array in para rs_dims has to
!>                  be initialized
!>           output: pw_grid
!>
!>      pw_grid_change : updates g-vectors after a change of the box
!> \par History
!>      JGH (20-12-2000) : Adapted for parallel use
!>      JGH (07-02-2001) : Added constructor and destructor routines
!>      JGH (21-02-2003) : Generalized reference grid concept
!>      JGH (19-11-2007) : Refactoring and modularization
!>      JGH (21-12-2007) : pw_grid_setup refactoring
!> \author apsi
!>      CJM
! **************************************************************************************************
MODULE pw_grids
   USE ISO_C_BINDING,                   ONLY: C_F_POINTER,&
                                              C_LOC,&
                                              C_PTR,&
                                              C_SIZE_T
   USE kinds,                           ONLY: dp,&
                                              int_8,&
                                              int_size
   USE mathconstants,                   ONLY: twopi
   USE mathlib,                         ONLY: det_3x3,&
                                              inv_3x3
   USE message_passing,                 ONLY: mp_comm_self,&
                                              mp_comm_type,&
                                              mp_dims_create
   USE offload_api,                     ONLY: offload_activate_chosen_device,&
                                              offload_free_pinned_mem,&
                                              offload_malloc_pinned_mem
   USE pw_grid_info,                    ONLY: pw_find_cutoff,&
                                              pw_grid_bounds_from_n,&
                                              pw_grid_init_setup
   USE pw_grid_types,                   ONLY: FULLSPACE,&
                                              HALFSPACE,&
                                              PW_MODE_DISTRIBUTED,&
                                              PW_MODE_LOCAL,&
                                              map_pn,&
                                              pw_grid_type
   USE util,                            ONLY: get_limit,&
                                              sort
!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_in_parallel
#include "../base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE
   PUBLIC :: pw_grid_create, pw_grid_retain, pw_grid_release
   PUBLIC :: get_pw_grid_info, pw_grid_compare
   PUBLIC :: pw_grid_change

   INTEGER :: grid_tag = 0
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'pw_grids'

   ! Distribution in g-space can be
   INTEGER, PARAMETER, PUBLIC               :: do_pw_grid_blocked_false = 0, &
                                               do_pw_grid_blocked_true = 1, &
                                               do_pw_grid_blocked_free = 2

   INTERFACE pw_grid_create
      MODULE PROCEDURE pw_grid_create_local
      MODULE PROCEDURE pw_grid_create_extended
   END INTERFACE

CONTAINS

! **************************************************************************************************
!> \brief Initialize a PW grid with bounds only (used by some routines)
!> \param pw_grid ...
!> \param bounds ...
!> \par History
!>      JGH (21-Feb-2003) : initialize pw_grid%reference
!> \author JGH (7-Feb-2001) & fawzi
! **************************************************************************************************
   SUBROUTINE pw_grid_create_local(pw_grid, bounds)

      TYPE(pw_grid_type), POINTER                        :: pw_grid
      INTEGER, DIMENSION(2, 3), INTENT(IN)               :: bounds

      INTEGER, DIMENSION(2)                              :: rs_dims

      CPASSERT(.NOT. ASSOCIATED(pw_grid))
      ALLOCATE (pw_grid)
      pw_grid%bounds = bounds
      pw_grid%bounds_local = bounds
      pw_grid%npts = bounds(2, :) - bounds(1, :) + 1
      pw_grid%npts_local = pw_grid%npts
      pw_grid%ngpts = PRODUCT(INT(pw_grid%npts, KIND=int_8))
      pw_grid%ngpts_cut = pw_grid%ngpts
      pw_grid%ngpts_local = PRODUCT(pw_grid%npts)
      pw_grid%ngpts_cut_local = pw_grid%ngpts_local
      pw_grid%grid_span = FULLSPACE
      pw_grid%para%mode = PW_MODE_LOCAL
      pw_grid%reference = 0
      pw_grid%ref_count = 1
      NULLIFY (pw_grid%g)
      NULLIFY (pw_grid%gsq)
      NULLIFY (pw_grid%g_hatmap)
      NULLIFY (pw_grid%gidx)
      NULLIFY (pw_grid%grays)

      ! assign a unique tag to this grid
      grid_tag = grid_tag + 1
      pw_grid%id_nr = grid_tag

      ! parallel info
      rs_dims = 1
      CALL pw_grid%para%group%create(mp_comm_self, 2, rs_dims)
      IF (pw_grid%para%group%num_pe > 1) THEN
         pw_grid%para%mode = PW_MODE_DISTRIBUTED
      ELSE
         pw_grid%para%mode = PW_MODE_LOCAL
      END IF

   END SUBROUTINE pw_grid_create_local

! **************************************************************************************************
!> \brief Check if two pw_grids are equal
!> \param grida ...
!> \param gridb ...
!> \return ...
!> \par History
!>      none
!> \author JGH (14-Feb-2001)
! **************************************************************************************************
   FUNCTION pw_grid_compare(grida, gridb) RESULT(equal)

      TYPE(pw_grid_type), INTENT(IN)                     :: grida, gridb
      LOGICAL                                            :: equal

!------------------------------------------------------------------------------

      IF (grida%id_nr == gridb%id_nr) THEN
         equal = .TRUE.
      ELSE
         ! for the moment all grids with different identifiers are considered as not equal
         ! later we can get this more relaxed
         equal = .FALSE.
      END IF

   END FUNCTION pw_grid_compare

! **************************************************************************************************
!> \brief Access to information stored in the pw_grid_type
!> \param pw_grid ...
!> \param id_nr ...
!> \param mode ...
!> \param vol ...
!> \param dvol ...
!> \param npts ...
!> \param ngpts ...
!> \param ngpts_cut ...
!> \param dr ...
!> \param cutoff ...
!> \param orthorhombic ...
!> \param gvectors ...
!> \param gsquare ...
!> \par History
!>      none
!> \author JGH (17-Nov-2007)
! **************************************************************************************************
   SUBROUTINE get_pw_grid_info(pw_grid, id_nr, mode, vol, dvol, npts, ngpts, &
                               ngpts_cut, dr, cutoff, orthorhombic, gvectors, gsquare)

      TYPE(pw_grid_type), INTENT(IN)                     :: pw_grid
      INTEGER, INTENT(OUT), OPTIONAL                     :: id_nr, mode
      REAL(dp), INTENT(OUT), OPTIONAL                    :: vol, dvol
      INTEGER, DIMENSION(3), INTENT(OUT), OPTIONAL       :: npts
      INTEGER(int_8), INTENT(OUT), OPTIONAL              :: ngpts, ngpts_cut
      REAL(dp), DIMENSION(3), INTENT(OUT), OPTIONAL      :: dr
      REAL(dp), INTENT(OUT), OPTIONAL                    :: cutoff
      LOGICAL, INTENT(OUT), OPTIONAL                     :: orthorhombic
      REAL(dp), DIMENSION(:, :), OPTIONAL, POINTER       :: gvectors
      REAL(dp), DIMENSION(:), OPTIONAL, POINTER          :: gsquare

      CPASSERT(pw_grid%ref_count > 0)

      IF (PRESENT(id_nr)) id_nr = pw_grid%id_nr
      IF (PRESENT(mode)) mode = pw_grid%para%mode
      IF (PRESENT(vol)) vol = pw_grid%vol
      IF (PRESENT(dvol)) dvol = pw_grid%dvol
      IF (PRESENT(npts)) npts(1:3) = pw_grid%npts(1:3)
      IF (PRESENT(ngpts)) ngpts = pw_grid%ngpts
      IF (PRESENT(ngpts_cut)) ngpts_cut = pw_grid%ngpts_cut
      IF (PRESENT(dr)) dr = pw_grid%dr
      IF (PRESENT(cutoff)) cutoff = pw_grid%cutoff
      IF (PRESENT(orthorhombic)) orthorhombic = pw_grid%orthorhombic
      IF (PRESENT(gvectors)) gvectors => pw_grid%g
      IF (PRESENT(gsquare)) gsquare => pw_grid%gsq

   END SUBROUTINE get_pw_grid_info

! **************************************************************************************************
!> \brief Set some information stored in the pw_grid_type
!> \param pw_grid ...
!> \param grid_span ...
!> \param npts ...
!> \param bounds ...
!> \param cutoff ...
!> \param spherical ...
!> \par History
!>      none
!> \author JGH (19-Nov-2007)
! **************************************************************************************************
   SUBROUTINE set_pw_grid_info(pw_grid, grid_span, npts, bounds, cutoff, spherical)

      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid
      INTEGER, INTENT(in), OPTIONAL                      :: grid_span
      INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL        :: npts
      INTEGER, DIMENSION(2, 3), INTENT(IN), OPTIONAL     :: bounds
      REAL(KIND=dp), INTENT(IN), OPTIONAL                :: cutoff
      LOGICAL, INTENT(IN), OPTIONAL                      :: spherical

      CPASSERT(pw_grid%ref_count > 0)

      IF (PRESENT(grid_span)) THEN
         pw_grid%grid_span = grid_span
      END IF
      IF (PRESENT(bounds) .AND. PRESENT(npts)) THEN
         pw_grid%bounds = bounds
         pw_grid%npts = npts
         CPASSERT(ALL(npts == bounds(2, :) - bounds(1, :) + 1))
      ELSE IF (PRESENT(bounds)) THEN
         pw_grid%bounds = bounds
         pw_grid%npts = bounds(2, :) - bounds(1, :) + 1
      ELSE IF (PRESENT(npts)) THEN
         pw_grid%npts = npts
         pw_grid%bounds = pw_grid_bounds_from_n(npts)
      END IF
      IF (PRESENT(cutoff)) THEN
         pw_grid%cutoff = cutoff
         IF (PRESENT(spherical)) THEN
            pw_grid%spherical = spherical
         ELSE
            pw_grid%spherical = .FALSE.
         END IF
      END IF

   END SUBROUTINE set_pw_grid_info

! **************************************************************************************************
!> \brief sets up a pw_grid
!> \param pw_grid ...
!> \param mp_comm ...
!> \param cell_hmat ...
!> \param grid_span ...
!> \param cutoff ...
!> \param bounds ...
!> \param bounds_local ...
!> \param npts ...
!> \param spherical ...
!> \param odd ...
!> \param fft_usage ...
!> \param ncommensurate ...
!> \param icommensurate ...
!> \param blocked ...
!> \param ref_grid ...
!> \param rs_dims ...
!> \param iounit ...
!> \author JGH (21-Dec-2007)
!> \note
!>      this is the function that should be used in the future
! **************************************************************************************************
   SUBROUTINE pw_grid_create_extended(pw_grid, mp_comm, cell_hmat, grid_span, cutoff, bounds, bounds_local, npts, &
                                      spherical, odd, fft_usage, ncommensurate, icommensurate, blocked, ref_grid, &
                                      rs_dims, iounit)

      TYPE(pw_grid_type), POINTER                        :: pw_grid
      REAL(KIND=dp), DIMENSION(3, 3), INTENT(IN)         :: cell_hmat

      CLASS(mp_comm_type), INTENT(IN) :: mp_comm
      INTEGER, INTENT(in), OPTIONAL                      :: grid_span
      REAL(KIND=dp), INTENT(IN), OPTIONAL                :: cutoff
      INTEGER, DIMENSION(2, 3), INTENT(IN), OPTIONAL     :: bounds, bounds_local
      INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL        :: npts
      LOGICAL, INTENT(in), OPTIONAL                      :: spherical, odd, fft_usage
      INTEGER, INTENT(in), OPTIONAL                      :: ncommensurate, icommensurate, blocked
      TYPE(pw_grid_type), INTENT(IN), OPTIONAL           :: ref_grid
      INTEGER, DIMENSION(2), INTENT(in), OPTIONAL        :: rs_dims
      INTEGER, INTENT(in), OPTIONAL                      :: iounit

      CHARACTER(len=*), PARAMETER                        :: routineN = 'pw_grid_setup'

      INTEGER                                            :: handle, my_icommensurate, &
                                                            my_ncommensurate
      INTEGER, DIMENSION(3)                              :: n
      LOGICAL                                            :: my_fft_usage, my_odd, my_spherical
      REAL(KIND=dp)                                      :: cell_deth, my_cutoff
      REAL(KIND=dp), DIMENSION(3, 3)                     :: cell_h_inv

      CALL timeset(routineN, handle)

      CPASSERT(.NOT. ASSOCIATED(pw_grid))
      ALLOCATE (pw_grid)
      pw_grid%bounds = 0
      pw_grid%cutoff = 0.0_dp
      pw_grid%grid_span = FULLSPACE
      pw_grid%para%mode = PW_MODE_LOCAL
      pw_grid%reference = 0
      pw_grid%ref_count = 1
      NULLIFY (pw_grid%g)
      NULLIFY (pw_grid%gsq)
      NULLIFY (pw_grid%g_hatmap)
      NULLIFY (pw_grid%gidx)
      NULLIFY (pw_grid%grays)

      ! assign a unique tag to this grid
      grid_tag = grid_tag + 1
      pw_grid%id_nr = grid_tag

      ! parallel info
      IF (mp_comm%num_pe > 1) THEN
         pw_grid%para%mode = PW_MODE_DISTRIBUTED
      ELSE
         pw_grid%para%mode = PW_MODE_LOCAL
      END IF

      cell_deth = ABS(det_3x3(cell_hmat))
      IF (cell_deth < 1.0E-10_dp) THEN
         CALL cp_abort(__LOCATION__, &
                       "An invalid set of cell vectors was specified. "// &
                       "The determinant det(h) is too small")
      END IF
      cell_h_inv = inv_3x3(cell_hmat)

      IF (PRESENT(grid_span)) THEN
         CALL set_pw_grid_info(pw_grid, grid_span=grid_span)
      END IF

      IF (PRESENT(spherical)) THEN
         my_spherical = spherical
      ELSE
         my_spherical = .FALSE.
      END IF

      IF (PRESENT(odd)) THEN
         my_odd = odd
      ELSE
         my_odd = .FALSE.
      END IF

      IF (PRESENT(fft_usage)) THEN
         my_fft_usage = fft_usage
      ELSE
         my_fft_usage = .FALSE.
      END IF

      IF (PRESENT(ncommensurate)) THEN
         my_ncommensurate = ncommensurate
         IF (PRESENT(icommensurate)) THEN
            my_icommensurate = icommensurate
         ELSE
            my_icommensurate = MIN(1, ncommensurate)
         END IF
      ELSE
         my_ncommensurate = 0
         my_icommensurate = 1
      END IF

      IF (PRESENT(bounds)) THEN
         IF (PRESENT(cutoff)) THEN
            CALL set_pw_grid_info(pw_grid, bounds=bounds, cutoff=cutoff, &
                                  spherical=my_spherical)
         ELSE
            n = bounds(2, :) - bounds(1, :) + 1
            my_cutoff = pw_find_cutoff(n, cell_h_inv)
            my_cutoff = 0.5_dp*my_cutoff*my_cutoff
            CALL set_pw_grid_info(pw_grid, bounds=bounds, cutoff=my_cutoff, &
                                  spherical=my_spherical)
         END IF
      ELSE IF (PRESENT(npts)) THEN
         n = npts
         IF (PRESENT(cutoff)) THEN
            my_cutoff = cutoff
         ELSE
            my_cutoff = pw_find_cutoff(npts, cell_h_inv)
            my_cutoff = 0.5_dp*my_cutoff*my_cutoff
         END IF
         IF (my_fft_usage) THEN
            n = pw_grid_init_setup(cell_hmat, cutoff=my_cutoff, &
                                   spherical=my_spherical, odd=my_odd, fft_usage=my_fft_usage, &
                                   ncommensurate=my_ncommensurate, icommensurate=my_icommensurate, &
                                   ref_grid=ref_grid, n_orig=n)
         END IF
         CALL set_pw_grid_info(pw_grid, npts=n, cutoff=my_cutoff, &
                               spherical=my_spherical)
      ELSE IF (PRESENT(cutoff)) THEN
         n = pw_grid_init_setup(cell_hmat, cutoff=cutoff, &
                                spherical=my_spherical, odd=my_odd, fft_usage=my_fft_usage, &
                                ncommensurate=my_ncommensurate, icommensurate=my_icommensurate, &
                                ref_grid=ref_grid)
         CALL set_pw_grid_info(pw_grid, npts=n, cutoff=cutoff, &
                               spherical=my_spherical)
      ELSE
         CPABORT("BOUNDS, NPTS or CUTOFF have to be specified")
      END IF

      CALL pw_grid_setup_internal(cell_hmat, cell_h_inv, cell_deth, pw_grid, mp_comm, bounds_local=bounds_local, &
                                  blocked=blocked, ref_grid=ref_grid, rs_dims=rs_dims, iounit=iounit)

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      CALL pw_grid_create_ghatmap(pw_grid)
#endif

      CALL timestop(handle)

   END SUBROUTINE pw_grid_create_extended

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
! **************************************************************************************************
!> \brief sets up a combined index for CUDA gather and scatter
!> \param pw_grid ...
!> \author Gloess Andreas (xx-Dec-2012)
! **************************************************************************************************
   SUBROUTINE pw_grid_create_ghatmap(pw_grid)

      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid

      CHARACTER(len=*), PARAMETER :: routineN = 'pw_grid_create_ghatmap'

      INTEGER                                            :: gpt, handle, j, l, m, mn, n, ngpts_l, &
                                                            nmap, nrow, span

      CALL timeset(routineN, handle)

      ! some checks
      CPASSERT(pw_grid%ref_count > 0)

      ! mapping of map_x( g_hat(i,j)) to g_hatmap
      ! the second index is for switching from gather(1) to scatter(2)
      IF (pw_grid%para%mode /= PW_MODE_DISTRIBUTED) THEN
         ! As below, in parallel: every row is initialized to minus one and then the entries
         ! of the G-vectors are set.
         ngpts_l = SIZE(pw_grid%gsq)
         nrow = SIZE(pw_grid%g_hatmap, 1)
         nmap = SIZE(pw_grid%g_hatmap, 2)
         span = pw_grid%grid_span
!$OMP PARALLEL DO DEFAULT(NONE) SHARED(ngpts_l, nmap, nrow, pw_grid, span) PRIVATE(gpt, j, l, m, n)
         DO gpt = 1, nrow
            DO j = 1, nmap
               pw_grid%g_hatmap(gpt, j) = -1
            END DO
            IF (gpt <= ngpts_l) THEN
               l = pw_grid%mapl%pos(pw_grid%g_hat(1, gpt))
               m = pw_grid%mapm%pos(pw_grid%g_hat(2, gpt))
               n = pw_grid%mapn%pos(pw_grid%g_hat(3, gpt))
               !ATTENTION: C-mapping [start-index=0] !!!!
               !ATTENTION: potential integer overflow !!!!
               pw_grid%g_hatmap(gpt, 1) = l + pw_grid%npts(1)*(m + pw_grid%npts(2)*n)
               IF (span == HALFSPACE) THEN
                  l = pw_grid%mapl%neg(pw_grid%g_hat(1, gpt))
                  m = pw_grid%mapm%neg(pw_grid%g_hat(2, gpt))
                  n = pw_grid%mapn%neg(pw_grid%g_hat(3, gpt))
                  pw_grid%g_hatmap(gpt, 2) = l + pw_grid%npts(1)*(m + pw_grid%npts(2)*n)
               END IF
            END IF
         END DO
!$OMP END PARALLEL DO
         CALL timestop(handle)
         RETURN
      END IF

      ASSOCIATE (g_hat => pw_grid%g_hat, g_hatmap => pw_grid%g_hatmap, pmapl => pw_grid%mapl%pos, &
                 pmapm => pw_grid%mapm%pos, pmapn => pw_grid%mapn%pos, nmapl => pw_grid%mapl%neg, &
                 nmapm => pw_grid%mapm%neg, nmapn => pw_grid%mapn%neg, ngpts => SIZE(pw_grid%gsq), &
                 npts => pw_grid%npts, yzq => pw_grid%para%yzq)
         ! initialize map array to minus one, to guarantee memory
         ! range checking errors in CUDA part (just to be sure)
         g_hatmap(:, :) = -1
         IF (pw_grid%para%mode /= PW_MODE_DISTRIBUTED) THEN
            DO gpt = 1, ngpts
               l = pmapl(g_hat(1, gpt))
               m = pmapm(g_hat(2, gpt))
               n = pmapn(g_hat(3, gpt))
               !ATTENTION: C-mapping [start-index=0] !!!!
               !ATTENTION: potential integer overflow !!!!
               g_hatmap(gpt, 1) = l + npts(1)*(m + npts(2)*n)
            END DO
            IF (pw_grid%grid_span == HALFSPACE) THEN
               DO gpt = 1, ngpts
                  l = nmapl(g_hat(1, gpt))
                  m = nmapm(g_hat(2, gpt))
                  n = nmapn(g_hat(3, gpt))
                  !ATTENTION: C-mapping [start-index=0] !!!!
                  !ATTENTION: potential integer overflow !!!!
                  g_hatmap(gpt, 2) = l + npts(1)*(m + npts(2)*n)
               END DO
            END IF
         ELSE
            DO gpt = 1, ngpts
               l = pmapl(g_hat(1, gpt))
               m = pmapm(g_hat(2, gpt)) + 1
               n = pmapn(g_hat(3, gpt)) + 1
               !ATTENTION: C-mapping [start-index=0] !!!!
               !ATTENTION: potential integer overflow !!!!
               mn = yzq(m, n) - 1
               g_hatmap(gpt, 1) = l + npts(1)*mn
            END DO
            IF (pw_grid%grid_span == HALFSPACE) THEN
               DO gpt = 1, ngpts
                  l = nmapl(g_hat(1, gpt))
                  m = nmapm(g_hat(2, gpt)) + 1
                  n = nmapn(g_hat(3, gpt)) + 1
                  !ATTENTION: C-mapping [start-index=0] !!!!
                  !ATTENTION: potential integer overflow !!!!
                  mn = yzq(m, n) - 1
                  g_hatmap(gpt, 2) = l + npts(1)*mn
               END DO
            END IF
         END IF
      END ASSOCIATE

      CALL timestop(handle)

   END SUBROUTINE pw_grid_create_ghatmap
#endif

! **************************************************************************************************
!> \brief sets up a pw_grid, needs valid bounds as input, it is up to you to
!>      make sure of it using pw_grid_bounds_from_n
!> \param cell_hmat ...
!> \param cell_h_inv ...
!> \param cell_deth ...
!> \param pw_grid ...
!> \param mp_comm ...
!> \param bounds_local ...
!> \param blocked ...
!> \param ref_grid ...
!> \param rs_dims ...
!> \param iounit ...
!> \par History
!>      JGH (20-Dec-2000) : Adapted for parallel use
!>      JGH (28-Feb-2001) : New optional argument fft_usage
!>      JGH (21-Mar-2001) : Reference grid code
!>      JGH (21-Mar-2001) : New optional argument symm_usage
!>      JGH (22-Mar-2001) : Simplify group assignment (mp_comm_dup)
!>      JGH (21-May-2002) : Remove orthorhombic keyword (code is fast enough)
!>      JGH (19-Feb-2003) : Negative cutoff can be used for non-spheric grids
!>      Joost VandeVondele (Feb-2004) : optionally generate pw grids that are commensurate in rs
!>      JGH (18-Dec-2007) : Refactoring
!> \author fawzi
!> \note
!>      this is the function that should be used in the future
! **************************************************************************************************
   SUBROUTINE pw_grid_setup_internal(cell_hmat, cell_h_inv, cell_deth, pw_grid, mp_comm, bounds_local, &
                                     blocked, ref_grid, rs_dims, iounit)
      REAL(KIND=dp), DIMENSION(3, 3), INTENT(IN)         :: cell_hmat, cell_h_inv
      REAL(KIND=dp), INTENT(IN)                          :: cell_deth
      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid

      CLASS(mp_comm_type), INTENT(IN) :: mp_comm
      INTEGER, DIMENSION(2, 3), INTENT(IN), OPTIONAL     :: bounds_local
      INTEGER, INTENT(in), OPTIONAL                      :: blocked
      TYPE(pw_grid_type), INTENT(in), OPTIONAL           :: ref_grid
      INTEGER, DIMENSION(2), INTENT(in), OPTIONAL        :: rs_dims
      INTEGER, INTENT(in), OPTIONAL                      :: iounit

      CHARACTER(len=*), PARAMETER :: routineN = 'pw_grid_setup_internal'

      INTEGER                                            :: handle, n(3)
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: yz_mask
      REAL(KIND=dp)                                      :: ecut

!------------------------------------------------------------------------------

      CALL timeset(routineN, handle)

      CPASSERT(pw_grid%ref_count > 0)

      ! set pointer to possible reference grid
      IF (PRESENT(ref_grid)) THEN
         pw_grid%reference = ref_grid%id_nr
      END IF

      IF (pw_grid%spherical) THEN
         ecut = pw_grid%cutoff
      ELSE
         ecut = 1.e10_dp
      END IF

      n(:) = pw_grid%npts(:)

      ! Find the number of grid points
      ! yz_mask counts the number of g-vectors orthogonal to the yz plane
      ! the indices in yz_mask are from -n/2 .. n/2 shifted by n/2 + 1
      ! these are not mapped indices !
      ALLOCATE (yz_mask(n(2), n(3)))
      CALL pw_grid_count(cell_h_inv, pw_grid, mp_comm, ecut, yz_mask)

      ! Check if reference grid is compatible
      IF (PRESENT(ref_grid)) THEN
         CPASSERT(pw_grid%para%mode == ref_grid%para%mode)
         CPASSERT(pw_grid%grid_span == ref_grid%grid_span)
         CPASSERT(pw_grid%spherical .EQV. ref_grid%spherical)
      END IF

      ! Distribute grid
      CALL pw_grid_distribute(pw_grid, mp_comm, yz_mask, bounds_local=bounds_local, ref_grid=ref_grid, blocked=blocked, &
                              rs_dims=rs_dims)

      ! Allocate the grid fields
      CALL pw_grid_allocate(pw_grid, pw_grid%ngpts_cut_local, &
                            pw_grid%bounds)

      ! Fill in the grid structure
      CALL pw_grid_assign(cell_h_inv, pw_grid, ecut)

      ! Sort g vector wrt length (only local for each processor)
      CALL pw_grid_sort(pw_grid, ref_grid)

      CALL pw_grid_remap(pw_grid, yz_mask)

      DEALLOCATE (yz_mask)

      CALL cell2grid(cell_hmat, cell_h_inv, cell_deth, pw_grid)
      !
      ! Output: All the information of this grid type
      !

      IF (PRESENT(iounit)) THEN
         CALL pw_grid_print(pw_grid, iounit)
      END IF

      CALL timestop(handle)

   END SUBROUTINE pw_grid_setup_internal

! **************************************************************************************************
!> \brief Helper routine used by pw_grid_setup_internal and pw_grid_change
!> \param cell_hmat ...
!> \param cell_h_inv ...
!> \param cell_deth ...
!> \param pw_grid ...
!> \par History moved common code into new subroutine
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE cell2grid(cell_hmat, cell_h_inv, cell_deth, pw_grid)
      REAL(KIND=dp), DIMENSION(3, 3), INTENT(IN)         :: cell_hmat, cell_h_inv
      REAL(KIND=dp), INTENT(IN)                          :: cell_deth
      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid

      pw_grid%vol = ABS(cell_deth)
      pw_grid%dvol = pw_grid%vol/REAL(pw_grid%ngpts, KIND=dp)
      pw_grid%dr(1) = SQRT(SUM(cell_hmat(:, 1)**2)) &
                      /REAL(pw_grid%npts(1), KIND=dp)
      pw_grid%dr(2) = SQRT(SUM(cell_hmat(:, 2)**2)) &
                      /REAL(pw_grid%npts(2), KIND=dp)
      pw_grid%dr(3) = SQRT(SUM(cell_hmat(:, 3)**2)) &
                      /REAL(pw_grid%npts(3), KIND=dp)
      pw_grid%dh(:, 1) = cell_hmat(:, 1)/REAL(pw_grid%npts(1), KIND=dp)
      pw_grid%dh(:, 2) = cell_hmat(:, 2)/REAL(pw_grid%npts(2), KIND=dp)
      pw_grid%dh(:, 3) = cell_hmat(:, 3)/REAL(pw_grid%npts(3), KIND=dp)
      pw_grid%dh_inv(1, :) = cell_h_inv(1, :)*REAL(pw_grid%npts(1), KIND=dp)
      pw_grid%dh_inv(2, :) = cell_h_inv(2, :)*REAL(pw_grid%npts(2), KIND=dp)
      pw_grid%dh_inv(3, :) = cell_h_inv(3, :)*REAL(pw_grid%npts(3), KIND=dp)

      IF ((cell_hmat(1, 2) == 0.0_dp) .AND. (cell_hmat(1, 3) == 0.0_dp) .AND. &
          (cell_hmat(2, 1) == 0.0_dp) .AND. (cell_hmat(2, 3) == 0.0_dp) .AND. &
          (cell_hmat(3, 1) == 0.0_dp) .AND. (cell_hmat(3, 2) == 0.0_dp)) THEN
         pw_grid%orthorhombic = .TRUE.
      ELSE
         pw_grid%orthorhombic = .FALSE.
      END IF
   END SUBROUTINE cell2grid

! **************************************************************************************************
!> \brief Output of information on pw_grid
!> \param pw_grid ...
!> \param info ...
!> \author JGH[18-05-2007] from earlier versions
! **************************************************************************************************
   SUBROUTINE pw_grid_print(pw_grid, info)

      TYPE(pw_grid_type), INTENT(IN)                     :: pw_grid
      INTEGER, INTENT(IN)                                :: info

      INTEGER                                            :: i
      INTEGER(KIND=int_8)                                :: n(3)
      REAL(KIND=dp)                                      :: rv(3, 3)

!------------------------------------------------------------------------------
!
! Output: All the information of this grid type
!

      IF (pw_grid%para%mode == PW_MODE_LOCAL) THEN
         IF (info > 0) THEN
            WRITE (info, '(/,A,T71,I10)') &
               " PW_GRID| Information for grid number ", pw_grid%id_nr
            IF (pw_grid%reference > 0) THEN
               WRITE (info, '(A,T71,I10)') &
                  " PW_GRID| Number of the reference grid ", pw_grid%reference
            END IF
            WRITE (info, '(" PW_GRID| Cutoff [a.u.]",T71,f10.1)') pw_grid%cutoff
            IF (pw_grid%spherical) THEN
               WRITE (info, '(A,T78,A)') " PW_GRID| spherical cutoff: ", "YES"
               WRITE (info, '(A,T71,I10)') " PW_GRID| Grid points within cutoff", &
                  pw_grid%ngpts_cut
            ELSE
               WRITE (info, '(A,T78,A)') " PW_GRID| spherical cutoff: ", " NO"
            END IF
            DO i = 1, 3
               WRITE (info, '(A,I3,T30,2I8,T62,A,T71,I10)') " PW_GRID|   Bounds ", &
                  i, pw_grid%bounds(1, I), pw_grid%bounds(2, I), &
                  "Points:", pw_grid%npts(I)
            END DO
            WRITE (info, '(A,G12.4,T50,A,T67,F14.4)') &
               " PW_GRID| Volume element (a.u.^3)", &
               pw_grid%dvol, " Volume (a.u.^3) :", pw_grid%vol
            IF (pw_grid%grid_span == HALFSPACE) THEN
               WRITE (info, '(A,T72,A)') " PW_GRID| Grid span", "HALFSPACE"
            ELSE
               WRITE (info, '(A,T72,A)') " PW_GRID| Grid span", "FULLSPACE"
            END IF
         END IF
      ELSE

         n(1) = pw_grid%ngpts_cut_local
         n(2) = pw_grid%ngpts_local
         CALL pw_grid%para%group%sum(n(1:2))
         n(3) = SUM(pw_grid%para%nyzray)
         rv(:, 1) = REAL(n, KIND=dp)/REAL(pw_grid%para%group%num_pe, KIND=dp)
         n(1) = pw_grid%ngpts_cut_local
         n(2) = pw_grid%ngpts_local
         CALL pw_grid%para%group%max(n(1:2))
         n(3) = MAXVAL(pw_grid%para%nyzray)
         rv(:, 2) = REAL(n, KIND=dp)
         n(1) = pw_grid%ngpts_cut_local
         n(2) = pw_grid%ngpts_local
         CALL pw_grid%para%group%min(n(1:2))
         n(3) = MINVAL(pw_grid%para%nyzray)
         rv(:, 3) = REAL(n, KIND=dp)

         IF (info > 0) THEN
            WRITE (info, '(/,A,T71,I10)') &
               " PW_GRID| Information for grid number ", pw_grid%id_nr
            IF (pw_grid%reference > 0) THEN
               WRITE (info, '(A,T71,I10)') &
                  " PW_GRID| Number of the reference grid ", pw_grid%reference
            END IF
            WRITE (info, '(A,T60,I10,A)') &
               " PW_GRID| Grid distributed over ", pw_grid%para%group%num_pe, &
               " processors"
            WRITE (info, '(A,T71,2I5)') &
               " PW_GRID| Real space group dimensions ", pw_grid%para%group%num_pe_cart
            IF (pw_grid%para%blocked) THEN
               WRITE (info, '(A,T78,A)') " PW_GRID| the grid is blocked: ", "YES"
            ELSE
               WRITE (info, '(A,T78,A)') " PW_GRID| the grid is blocked: ", " NO"
            END IF
            WRITE (info, '(" PW_GRID| Cutoff [a.u.]",T71,f10.1)') pw_grid%cutoff
            IF (pw_grid%spherical) THEN
               WRITE (info, '(A,T78,A)') " PW_GRID| spherical cutoff: ", "YES"
               WRITE (info, '(A,T71,I10)') " PW_GRID| Grid points within cutoff", &
                  pw_grid%ngpts_cut
            ELSE
               WRITE (info, '(A,T78,A)') " PW_GRID| spherical cutoff: ", " NO"
            END IF
            DO i = 1, 3
               WRITE (info, '(A,I3,T30,2I8,T62,A,T71,I10)') " PW_GRID|   Bounds ", &
                  i, pw_grid%bounds(1, I), pw_grid%bounds(2, I), &
                  "Points:", pw_grid%npts(I)
            END DO
            WRITE (info, '(A,G12.4,T50,A,T67,F14.4)') &
               " PW_GRID| Volume element (a.u.^3)", &
               pw_grid%dvol, " Volume (a.u.^3) :", pw_grid%vol
            IF (pw_grid%grid_span == HALFSPACE) THEN
               WRITE (info, '(A,T72,A)') " PW_GRID| Grid span", "HALFSPACE"
            ELSE
               WRITE (info, '(A,T72,A)') " PW_GRID| Grid span", "FULLSPACE"
            END IF
            WRITE (info, '(A,T48,A)') " PW_GRID|   Distribution", &
               "  Average         Max         Min"
            WRITE (info, '(A,T45,F12.1,2I12)') " PW_GRID|   G-Vectors", &
               rv(1, 1), NINT(rv(1, 2)), NINT(rv(1, 3))
            WRITE (info, '(A,T45,F12.1,2I12)') " PW_GRID|   G-Rays   ", &
               rv(3, 1), NINT(rv(3, 2)), NINT(rv(3, 3))
            WRITE (info, '(A,T45,F12.1,2I12)') " PW_GRID|   Real Space Points", &
               rv(2, 1), NINT(rv(2, 2)), NINT(rv(2, 3))
         END IF ! group head
      END IF ! local

   END SUBROUTINE pw_grid_print

! **************************************************************************************************
!> \brief Distribute grids in real and Fourier Space to the processors in group
!> \param pw_grid ...
!> \param mp_comm ...
!> \param yz_mask ...
!> \param bounds_local ...
!> \param ref_grid ...
!> \param blocked ...
!> \param rs_dims ...
!> \par History
!>      JGH (01-Mar-2001) optional reference grid
!>      JGH (22-May-2002) bug fix for pre_tag and HALFSPACE grids
!>      JGH (09-Sep-2003) reduce scaling for distribution
!> \author JGH (22-12-2000)
! **************************************************************************************************
   SUBROUTINE pw_grid_distribute(pw_grid, mp_comm, yz_mask, bounds_local, ref_grid, blocked, rs_dims)

      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid

      CLASS(mp_comm_type), INTENT(IN) :: mp_comm
      INTEGER, DIMENSION(:, :), INTENT(INOUT)            :: yz_mask
      INTEGER, DIMENSION(2, 3), INTENT(IN), OPTIONAL     :: bounds_local
      TYPE(pw_grid_type), INTENT(IN), OPTIONAL           :: ref_grid
      INTEGER, INTENT(IN), OPTIONAL                      :: blocked
      INTEGER, DIMENSION(2), INTENT(in), OPTIONAL        :: rs_dims

      CHARACTER(len=*), PARAMETER :: routineN = 'pw_grid_distribute'

      INTEGER                                            :: blocked_local, coor(2), gmax, handle, i, &
                                                            i1, i2, ip, ipl, ipp, itmp, j, l, lby, &
                                                            lbz, lo(2), m, n, np, ns, nx, ny, nz, &
                                                            rsd(2)
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: pemap
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: yz_index
      INTEGER, ALLOCATABLE, DIMENSION(:, :, :)           :: axis_dist_all
      INTEGER, DIMENSION(2)                              :: my_rs_dims
      INTEGER, DIMENSION(2, 3)                           :: axis_dist
      LOGICAL                                            :: blocking

!------------------------------------------------------------------------------

      CALL timeset(routineN, handle)

      lby = pw_grid%bounds(1, 2)
      lbz = pw_grid%bounds(1, 3)

      pw_grid%ngpts = PRODUCT(INT(pw_grid%npts, KIND=int_8))

      my_rs_dims = 0
      IF (PRESENT(rs_dims)) THEN
         my_rs_dims = rs_dims
      END IF

      IF (PRESENT(blocked)) THEN
         blocked_local = blocked
      ELSE
         blocked_local = do_pw_grid_blocked_free
      END IF

      pw_grid%para%blocked = .FALSE.

      IF (pw_grid%para%mode == PW_MODE_LOCAL) THEN

         pw_grid%para%ray_distribution = .FALSE.

         pw_grid%bounds_local = pw_grid%bounds
         pw_grid%npts_local = pw_grid%npts
         CPASSERT(pw_grid%ngpts_cut < HUGE(pw_grid%ngpts_cut_local))
         pw_grid%ngpts_cut_local = INT(pw_grid%ngpts_cut)
         CPASSERT(pw_grid%ngpts < HUGE(pw_grid%ngpts_local))
         pw_grid%ngpts_local = INT(pw_grid%ngpts)
         my_rs_dims = 1
         CALL pw_grid%para%group%create(mp_comm, 2, my_rs_dims)

         ALLOCATE (pw_grid%para%bo(2, 3, 0:0, 3))
         DO i = 1, 3
            pw_grid%para%bo(1, 1:3, 0, i) = 1
            pw_grid%para%bo(2, 1:3, 0, i) = pw_grid%npts(1:3)
         END DO

      ELSE

         !..find the real space distribution
         nx = pw_grid%npts(1)
         ny = pw_grid%npts(2)
         nz = pw_grid%npts(3)
         np = mp_comm%num_pe

         ! The user can specify 2 strictly positive indices => specific layout
         !                      1 strictly positive index   => the other is fixed by the number of CPUs
         !                      0 strictly positive indices => fully free distribution
         ! if fully free or the user request can not be fulfilled, we optimize heuristically ourselves by:
         !                      1) nx>np -> taking a plane distribution (/np,1/)
         !                      2) nx<np -> taking the most square distribution
         ! if blocking is free:
         !                      1) blocked=.FALSE. for plane distributions
         !                      2) blocked=.TRUE.  for non-plane distributions
         IF (ANY(my_rs_dims <= 0)) THEN
            IF (ALL(my_rs_dims <= 0)) THEN
               my_rs_dims = [0, 0]
            ELSE
               IF (my_rs_dims(1) > 0) THEN
                  my_rs_dims(2) = np/my_rs_dims(1)
               ELSE
                  my_rs_dims(1) = np/my_rs_dims(2)
               END IF
            END IF
         END IF
         ! reset if the distribution can not be fulfilled
         IF (PRODUCT(my_rs_dims) /= np) my_rs_dims = [0, 0]
         ! reset if the distribution can not be dealt with [1,np]
         IF (ALL(my_rs_dims == [1, np])) my_rs_dims = [0, 0]

         ! if [0,0] now, we can optimize it ourselves
         IF (ALL(my_rs_dims == [0, 0])) THEN
            ! only small grids have a chance to be 2d distributed
            IF (nx < np) THEN
               ! gives the most square looking distribution
               CALL mp_dims_create(np, my_rs_dims)
               ! we tend to like the first index being smaller than the second
               IF (my_rs_dims(1) > my_rs_dims(2)) THEN
                  itmp = my_rs_dims(1)
                  my_rs_dims(1) = my_rs_dims(2)
                  my_rs_dims(2) = itmp
               END IF
               ! but should avoid having the first index 1 in all cases
               IF (my_rs_dims(1) == 1) THEN
                  itmp = my_rs_dims(1)
                  my_rs_dims(1) = my_rs_dims(2)
                  my_rs_dims(2) = itmp
               END IF
            ELSE
               my_rs_dims = [np, 1]
            END IF
         END IF

         ! now fix the blocking if we have a choice
         SELECT CASE (blocked_local)
         CASE (do_pw_grid_blocked_false)
            blocking = .FALSE.
         CASE (do_pw_grid_blocked_true)
            blocking = .TRUE.
         CASE (do_pw_grid_blocked_free)
            IF (ALL(my_rs_dims == [np, 1])) THEN
               blocking = .FALSE.
            ELSE
               blocking = .TRUE.
            END IF
         CASE DEFAULT
            CPABORT("Unknown PW grid blocking")
         END SELECT

         !..create group for real space distribution
         CALL pw_grid%para%group%create(mp_comm, 2, my_rs_dims)

         IF (PRESENT(bounds_local)) THEN
            pw_grid%bounds_local = bounds_local
         ELSE
            lo = get_limit(nx, pw_grid%para%group%num_pe_cart(1), &
                           pw_grid%para%group%mepos_cart(1))
            pw_grid%bounds_local(:, 1) = lo + pw_grid%bounds(1, 1) - 1
            lo = get_limit(ny, pw_grid%para%group%num_pe_cart(2), &
                           pw_grid%para%group%mepos_cart(2))
            pw_grid%bounds_local(:, 2) = lo + pw_grid%bounds(1, 2) - 1
            pw_grid%bounds_local(:, 3) = pw_grid%bounds(:, 3)
         END IF

         pw_grid%npts_local(:) = pw_grid%bounds_local(2, :) &
                                 - pw_grid%bounds_local(1, :) + 1

         !..the third distribution is needed for the second step in the FFT
         ALLOCATE (pw_grid%para%bo(2, 3, 0:np - 1, 3))
         rsd = pw_grid%para%group%num_pe_cart

         IF (PRESENT(bounds_local)) THEN
            ! axis_dist tells what portion of 1 .. nx , 1 .. ny , 1 .. nz are in the current process
            DO i = 1, 3
               axis_dist(:, i) = bounds_local(:, i) - pw_grid%bounds(1, i) + 1
            END DO
            ALLOCATE (axis_dist_all(2, 3, np))
            CALL pw_grid%para%group%allgather(axis_dist, axis_dist_all)
            DO ip = 0, np - 1
               CALL pw_grid%para%group%coords(ip, coor)
               ! distribution xyZ
               pw_grid%para%bo(1:2, 1, ip, 1) = axis_dist_all(1:2, 1, ip + 1)
               pw_grid%para%bo(1:2, 2, ip, 1) = axis_dist_all(1:2, 2, ip + 1)
               pw_grid%para%bo(1, 3, ip, 1) = 1
               pw_grid%para%bo(2, 3, ip, 1) = nz
               ! distribution xYz
               pw_grid%para%bo(1:2, 1, ip, 2) = axis_dist_all(1:2, 1, ip + 1)
               pw_grid%para%bo(1, 2, ip, 2) = 1
               pw_grid%para%bo(2, 2, ip, 2) = ny
               pw_grid%para%bo(1:2, 3, ip, 2) = get_limit(nz, rsd(2), coor(2))
               ! distribution Xyz
               pw_grid%para%bo(1, 1, ip, 3) = 1
               pw_grid%para%bo(2, 1, ip, 3) = nx
               pw_grid%para%bo(1:2, 2, ip, 3) = get_limit(ny, rsd(1), coor(1))
               pw_grid%para%bo(1:2, 3, ip, 3) = get_limit(nz, rsd(2), coor(2))
            END DO
            DEALLOCATE (axis_dist_all)
         ELSE
            DO ip = 0, np - 1
               CALL pw_grid%para%group%coords(ip, coor)
               ! distribution xyZ
               pw_grid%para%bo(1:2, 1, ip, 1) = get_limit(nx, rsd(1), coor(1))
               pw_grid%para%bo(1:2, 2, ip, 1) = get_limit(ny, rsd(2), coor(2))
               pw_grid%para%bo(1, 3, ip, 1) = 1
               pw_grid%para%bo(2, 3, ip, 1) = nz
               ! distribution xYz
               pw_grid%para%bo(1:2, 1, ip, 2) = get_limit(nx, rsd(1), coor(1))
               pw_grid%para%bo(1, 2, ip, 2) = 1
               pw_grid%para%bo(2, 2, ip, 2) = ny
               pw_grid%para%bo(1:2, 3, ip, 2) = get_limit(nz, rsd(2), coor(2))
               ! distribution Xyz
               pw_grid%para%bo(1, 1, ip, 3) = 1
               pw_grid%para%bo(2, 1, ip, 3) = nx
               pw_grid%para%bo(1:2, 2, ip, 3) = get_limit(ny, rsd(1), coor(1))
               pw_grid%para%bo(1:2, 3, ip, 3) = get_limit(nz, rsd(2), coor(2))
            END DO
         END IF
         !..find the g space distribution
         pw_grid%ngpts_cut_local = 0

         ALLOCATE (pw_grid%para%nyzray(0:np - 1))

         ALLOCATE (pw_grid%para%yzq(ny, nz))

         IF (pw_grid%spherical .OR. pw_grid%grid_span == HALFSPACE &
             .OR. .NOT. blocking) THEN

            pw_grid%para%ray_distribution = .TRUE.

            pw_grid%para%yzq = -1
            IF (PRESENT(ref_grid)) THEN
               ! tag all vectors from the reference grid
               CALL pre_tag(pw_grid, yz_mask, ref_grid)
            END IF

            ! Round Robin distribution
            ! Processors 0 .. NP-1, NP-1 .. 0  get the largest remaining batch
            ! of g vectors in turn

            i1 = SIZE(yz_mask, 1)
            i2 = SIZE(yz_mask, 2)
            ALLOCATE (yz_index(2, i1*i2))
            CALL order_mask(yz_mask, yz_index)
            DO i = 1, i1*i2
               lo(1) = yz_index(1, i)
               lo(2) = yz_index(2, i)
               IF (lo(1)*lo(2) == 0) CYCLE
               gmax = yz_mask(lo(1), lo(2))
               IF (gmax == 0) CYCLE
               yz_mask(lo(1), lo(2)) = 0
               ip = MOD(i - 1, 2*np)
               IF (ip > np - 1) ip = 2*np - ip - 1
               IF (ip == pw_grid%para%group%mepos) THEN
                  pw_grid%ngpts_cut_local = pw_grid%ngpts_cut_local + gmax
               END IF
               pw_grid%para%yzq(lo(1), lo(2)) = ip
               IF (pw_grid%grid_span == HALFSPACE) THEN
                  m = -lo(1) - 2*lby + 2
                  n = -lo(2) - 2*lbz + 2
                  pw_grid%para%yzq(m, n) = ip
                  yz_mask(m, n) = 0
               END IF
            END DO

            DEALLOCATE (yz_index)

            ! Count the total number of rays on each processor
            pw_grid%para%nyzray = 0
            DO i = 1, nz
               DO j = 1, ny
                  ip = pw_grid%para%yzq(j, i)
                  IF (ip >= 0) pw_grid%para%nyzray(ip) = &
                     pw_grid%para%nyzray(ip) + 1
               END DO
            END DO

            ! Allocate mapping array (y:z, nray, nproc)
            ns = MAXVAL(pw_grid%para%nyzray(0:np - 1))
            ALLOCATE (pw_grid%para%yzp(2, ns, 0:np - 1))

            ! Fill mapping array, recalculate nyzray for convenience
            pw_grid%para%nyzray = 0
            DO i = 1, nz
               DO j = 1, ny
                  ip = pw_grid%para%yzq(j, i)
                  IF (ip >= 0) THEN
                     pw_grid%para%nyzray(ip) = &
                        pw_grid%para%nyzray(ip) + 1
                     ns = pw_grid%para%nyzray(ip)
                     pw_grid%para%yzp(1, ns, ip) = j
                     pw_grid%para%yzp(2, ns, ip) = i
                     IF (ip == pw_grid%para%group%mepos) THEN
                        pw_grid%para%yzq(j, i) = ns
                     ELSE
                        pw_grid%para%yzq(j, i) = -1
                     END IF
                  ELSE
                     pw_grid%para%yzq(j, i) = -2
                  END IF
               END DO
            END DO

            pw_grid%ngpts_local = PRODUCT(pw_grid%npts_local)

         ELSE
            !
            !  block distribution of g vectors, we do not have a spherical cutoff
            !

            pw_grid%para%blocked = .TRUE.
            pw_grid%para%ray_distribution = .FALSE.

            DO ip = 0, np - 1
               m = pw_grid%para%bo(2, 2, ip, 3) - &
                   pw_grid%para%bo(1, 2, ip, 3) + 1
               n = pw_grid%para%bo(2, 3, ip, 3) - &
                   pw_grid%para%bo(1, 3, ip, 3) + 1
               pw_grid%para%nyzray(ip) = n*m
            END DO

            ipl = pw_grid%para%group%mepos
            l = pw_grid%para%bo(2, 1, ipl, 3) - &
                pw_grid%para%bo(1, 1, ipl, 3) + 1
            m = pw_grid%para%bo(2, 2, ipl, 3) - &
                pw_grid%para%bo(1, 2, ipl, 3) + 1
            n = pw_grid%para%bo(2, 3, ipl, 3) - &
                pw_grid%para%bo(1, 3, ipl, 3) + 1
            pw_grid%ngpts_cut_local = l*m*n
            pw_grid%ngpts_local = pw_grid%ngpts_cut_local

            pw_grid%para%yzq = 0
            ny = pw_grid%para%bo(2, 2, ipl, 3) - &
                 pw_grid%para%bo(1, 2, ipl, 3) + 1
            DO n = pw_grid%para%bo(1, 3, ipl, 3), &
               pw_grid%para%bo(2, 3, ipl, 3)
               i = n - pw_grid%para%bo(1, 3, ipl, 3)
               DO m = pw_grid%para%bo(1, 2, ipl, 3), &
                  pw_grid%para%bo(2, 2, ipl, 3)
                  j = m - pw_grid%para%bo(1, 2, ipl, 3) + 1
                  pw_grid%para%yzq(m, n) = j + i*ny
               END DO
            END DO

            ! Allocate mapping array (y:z, nray, nproc)
            ns = MAXVAL(pw_grid%para%nyzray(0:np - 1))
            ALLOCATE (pw_grid%para%yzp(2, ns, 0:np - 1))
            pw_grid%para%yzp = 0

            ALLOCATE (pemap(0:np - 1))
            pemap = 0
            pemap(pw_grid%para%group%mepos) = pw_grid%para%group%mepos
            CALL pw_grid%para%group%sum(pemap)

            DO ip = 0, np - 1
               ipp = pemap(ip)
               ns = 0
               DO n = pw_grid%para%bo(1, 3, ipp, 3), &
                  pw_grid%para%bo(2, 3, ipp, 3)
                  i = n - pw_grid%bounds(1, 3) + 1
                  DO m = pw_grid%para%bo(1, 2, ipp, 3), &
                     pw_grid%para%bo(2, 2, ipp, 3)
                     j = m - pw_grid%bounds(1, 2) + 1
                     ns = ns + 1
                     pw_grid%para%yzp(1, ns, ip) = j
                     pw_grid%para%yzp(2, ns, ip) = i
                  END DO
               END DO
               CPASSERT(ns == pw_grid%para%nyzray(ip))
            END DO

            DEALLOCATE (pemap)

         END IF

      END IF

      ! pos_of_x(i) tells on which cpu pw%array(i,:,:) is located
      ! should be computable in principle, without the need for communication
      IF (pw_grid%para%mode == PW_MODE_DISTRIBUTED) THEN
         ALLOCATE (pw_grid%para%pos_of_x(pw_grid%bounds(1, 1):pw_grid%bounds(2, 1)))
         pw_grid%para%pos_of_x = 0
         pw_grid%para%pos_of_x(pw_grid%bounds_local(1, 1):pw_grid%bounds_local(2, 1)) = pw_grid%para%group%mepos
         CALL pw_grid%para%group%sum(pw_grid%para%pos_of_x)
      ELSE
         ! this should not be needed
         ALLOCATE (pw_grid%para%pos_of_x(pw_grid%bounds(1, 1):pw_grid%bounds(2, 1)))
         pw_grid%para%pos_of_x = 0
      END IF

      CALL timestop(handle)

   END SUBROUTINE pw_grid_distribute

! **************************************************************************************************
!> \brief ...
!> \param pw_grid ...
!> \param yz_mask ...
!> \param ref_grid ...
!> \par History
!>      - Fix mapping bug for pw_grid eqv to ref_grid (21.11.2019, MK)
! **************************************************************************************************
   SUBROUTINE pre_tag(pw_grid, yz_mask, ref_grid)

      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid
      INTEGER, DIMENSION(:, :), INTENT(INOUT)            :: yz_mask
      TYPE(pw_grid_type), INTENT(IN)                     :: ref_grid

      INTEGER                                            :: gmax, ig, ip, lby, lbz, my, mz, ny, nz, &
                                                            uby, ubz, y, yp, z, zp

      ny = ref_grid%npts(2)
      nz = ref_grid%npts(3)
      lby = pw_grid%bounds(1, 2)
      lbz = pw_grid%bounds(1, 3)
      uby = pw_grid%bounds(2, 2)
      ubz = pw_grid%bounds(2, 3)
      my = SIZE(yz_mask, 1)
      mz = SIZE(yz_mask, 2)

      ! loop over all processors and all g vectors yz lines on this processor
      DO ip = 0, ref_grid%para%group%num_pe - 1
         DO ig = 1, ref_grid%para%nyzray(ip)
            ! go from mapped coordinates to original coordinates
            ! 1, 2, ..., n-1, n -> 0, 1, ..., (n/2)-1, -(n/2), -(n/2)+1, ..., -2, -1
            y = ref_grid%para%yzp(1, ig, ip) - 1
            IF (y >= ny/2) y = y - ny
            z = ref_grid%para%yzp(2, ig, ip) - 1
            IF (z >= nz/2) z = z - nz
            ! check if this is inside the realm of the new grid
            IF (y < lby .OR. y > uby .OR. z < lbz .OR. z > ubz) CYCLE
            ! go to shifted coordinates
            y = y - lby + 1
            z = z - lbz + 1
            ! this tag is outside the cutoff range of the new grid
            IF (pw_grid%grid_span == HALFSPACE) THEN
               yp = -y - 2*lby + 2
               zp = -z - 2*lbz + 2
               ! if the reference grid is larger than the mirror point may be
               ! outside the new grid even if the original point is inside
               IF (yp < 1 .OR. yp > my .OR. zp < 1 .OR. zp > mz) CYCLE
               gmax = MAX(yz_mask(y, z), yz_mask(yp, zp))
               IF (gmax == 0) CYCLE
               yz_mask(y, z) = 0
               yz_mask(yp, zp) = 0
               pw_grid%para%yzq(y, z) = ip
               pw_grid%para%yzq(yp, zp) = ip
            ELSE
               gmax = yz_mask(y, z)
               IF (gmax == 0) CYCLE
               yz_mask(y, z) = 0
               pw_grid%para%yzq(y, z) = ip
            END IF
            IF (ip == pw_grid%para%group%mepos) THEN
               pw_grid%ngpts_cut_local = pw_grid%ngpts_cut_local + gmax
            END IF
         END DO
      END DO

   END SUBROUTINE pre_tag

! **************************************************************************************************
!> \brief ...
!> \param yz_mask ...
!> \param yz_index ...
! **************************************************************************************************
   PURE SUBROUTINE order_mask(yz_mask, yz_index)

      INTEGER, DIMENSION(:, :), INTENT(IN)               :: yz_mask
      INTEGER, DIMENSION(:, :), INTENT(OUT)              :: yz_index

      INTEGER                                            :: i1, i2, ic, icount, ii, im, jc, jj

!NB load balance
!------------------------------------------------------------------------------
!NB spiral out from origin, so that even if overall grid is full and
!NB block distributed, spherical cutoff still leads to good load
!NB balance in cp_ddapc_apply_CD

      i1 = SIZE(yz_mask, 1)
      i2 = SIZE(yz_mask, 2)
      yz_index = 0

      icount = 1
      ic = i1/2
      jc = i2/2
      ii = ic
      jj = jc
      IF (ii > 0 .AND. ii <= i1 .AND. jj > 0 .AND. jj <= i2) THEN
         IF (yz_mask(ii, jj) /= 0) THEN
            yz_index(1, icount) = ii
            yz_index(2, icount) = jj
            icount = icount + 1
         END IF
      END IF
      DO im = 1, MAX(ic + 1, jc + 1)
         ii = ic - im
         DO jj = jc - im, jc + im
            IF (ii > 0 .AND. ii <= i1 .AND. jj > 0 .AND. jj <= i2) THEN
               IF (yz_mask(ii, jj) /= 0) THEN
                  yz_index(1, icount) = ii
                  yz_index(2, icount) = jj
                  icount = icount + 1
               END IF
            END IF
         END DO
         ii = ic + im
         DO jj = jc - im, jc + im
            IF (ii > 0 .AND. ii <= i1 .AND. jj > 0 .AND. jj <= i2) THEN
               IF (yz_mask(ii, jj) /= 0) THEN
                  yz_index(1, icount) = ii
                  yz_index(2, icount) = jj
                  icount = icount + 1
               END IF
            END IF
         END DO
         jj = jc - im
         DO ii = ic - im + 1, ic + im - 1
            IF (ii > 0 .AND. ii <= i1 .AND. jj > 0 .AND. jj <= i2) THEN
               IF (yz_mask(ii, jj) /= 0) THEN
                  yz_index(1, icount) = ii
                  yz_index(2, icount) = jj
                  icount = icount + 1
               END IF
            END IF
         END DO
         jj = jc + im
         DO ii = ic - im + 1, ic + im - 1
            IF (ii > 0 .AND. ii <= i1 .AND. jj > 0 .AND. jj <= i2) THEN
               IF (yz_mask(ii, jj) /= 0) THEN
                  yz_index(1, icount) = ii
                  yz_index(2, icount) = jj
                  icount = icount + 1
               END IF
            END IF
         END DO
      END DO

   END SUBROUTINE order_mask
! **************************************************************************************************
!> \brief compute the length of g vectors
!> \param h_inv ...
!> \param length_x ...
!> \param length_y ...
!> \param length_z ...
!> \param length ...
!> \param l ...
!> \param m ...
!> \param n ...
! **************************************************************************************************
   PURE SUBROUTINE pw_vec_length(h_inv, length_x, length_y, length_z, length, l, m, n)

      REAL(KIND=dp), DIMENSION(3, 3), INTENT(IN)         :: h_inv
      REAL(KIND=dp), INTENT(OUT)                         :: length_x, length_y, length_z, length
      INTEGER, INTENT(IN)                                :: l, m, n

      length_x &
         = REAL(l, dp)*h_inv(1, 1) &
           + REAL(m, dp)*h_inv(2, 1) &
           + REAL(n, dp)*h_inv(3, 1)
      length_y &
         = REAL(l, dp)*h_inv(1, 2) &
           + REAL(m, dp)*h_inv(2, 2) &
           + REAL(n, dp)*h_inv(3, 2)
      length_z &
         = REAL(l, dp)*h_inv(1, 3) &
           + REAL(m, dp)*h_inv(2, 3) &
           + REAL(n, dp)*h_inv(3, 3)

      ! enforce strict zero-ness in this case (compiler optimization)
      IF (l == 0 .AND. m == 0 .AND. n == 0) THEN
         length_x = 0
         length_y = 0
         length_z = 0
      END IF

      length_x = length_x*twopi
      length_y = length_y*twopi
      length_z = length_z*twopi

      length = length_x**2 + length_y**2 + length_z**2

   END SUBROUTINE pw_vec_length

! **************************************************************************************************
!> \brief Count total number of g vectors
!> \param h_inv ...
!> \param pw_grid ...
!> \param mp_comm ...
!> \param cutoff ...
!> \param yz_mask ...
!> \par History
!>      JGH (22-12-2000) : Adapted for parallel use
!> \author apsi
!>      Christopher Mundy
! **************************************************************************************************
   SUBROUTINE pw_grid_count(h_inv, pw_grid, mp_comm, cutoff, yz_mask)

      REAL(KIND=dp), DIMENSION(3, 3)                     :: h_inv
      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid

      CLASS(mp_comm_type), INTENT(IN) :: mp_comm
      REAL(KIND=dp), INTENT(IN)                          :: cutoff
      INTEGER, DIMENSION(:, :), INTENT(OUT)              :: yz_mask

      INTEGER                                            :: l, m, mm, n, n_upperlimit, nlim(2), nn, &
                                                            span
      INTEGER(KIND=int_8)                                :: gpt
      INTEGER, DIMENSION(2, 3)                           :: bounds
      REAL(KIND=dp)                                      :: length, length_x, length_y, length_z

      bounds = pw_grid%bounds
      span = pw_grid%grid_span

      IF (span == HALFSPACE) THEN
         n_upperlimit = 0
      ELSE IF (span == FULLSPACE) THEN
         n_upperlimit = bounds(2, 3)
      ELSE
         CPABORT("No type set for the grid")
      END IF

      ! finds valid g-points within grid
      gpt = 0
      IF (pw_grid%para%mode == PW_MODE_LOCAL) THEN
         nlim(1) = bounds(1, 3)
         nlim(2) = n_upperlimit
      ELSE IF (pw_grid%para%mode == PW_MODE_DISTRIBUTED) THEN
         n = n_upperlimit - bounds(1, 3) + 1
         nlim = get_limit(n, mp_comm%num_pe, mp_comm%mepos)
         nlim = nlim + bounds(1, 3) - 1
      ELSE
         CPABORT("para % mode not specified")
      END IF

      ! every plane n updates only its own column of yz_mask
      yz_mask = 0
!$OMP PARALLEL DO DEFAULT(NONE) SHARED(bounds, cutoff, h_inv, nlim, span, yz_mask) &
!$OMP             PRIVATE(l, length, length_x, length_y, length_z, m, mm, n, nn) &
!$OMP             REDUCTION(+:gpt) SCHEDULE(DYNAMIC, 1)
      DO n = nlim(1), nlim(2)
         nn = n - bounds(1, 3) + 1
         DO m = bounds(1, 2), bounds(2, 2)
            mm = m - bounds(1, 2) + 1
            DO l = bounds(1, 1), bounds(2, 1)
               IF (span == HALFSPACE .AND. n == 0) THEN
                  IF ((m == 0 .AND. l > 0) .OR. (m > 0)) CYCLE
               END IF

               CALL pw_vec_length(h_inv, length_x, length_y, length_z, length, l, m, n)

               IF (0.5_dp*length <= cutoff) THEN
                  gpt = gpt + 1
                  yz_mask(mm, nn) = yz_mask(mm, nn) + 1
               END IF

            END DO
         END DO
      END DO
!$OMP END PARALLEL DO

      ! number of g-vectors for grid
      IF (pw_grid%para%mode == PW_MODE_DISTRIBUTED) THEN
         CALL mp_comm%sum(gpt)
         CALL mp_comm%sum(yz_mask)
      END IF
      pw_grid%ngpts_cut = gpt

   END SUBROUTINE pw_grid_count

! **************************************************************************************************
!> \brief Setup maps from 1d to 3d space
!> \param h_inv ...
!> \param pw_grid ...
!> \param cutoff ...
!> \par History
!>      JGH (29-12-2000) : Adapted for parallel use
!> \author apsi
!>      Christopher Mundy
! **************************************************************************************************
   SUBROUTINE pw_grid_assign(h_inv, pw_grid, cutoff)

      REAL(KIND=dp), DIMENSION(3, 3)                     :: h_inv
      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid
      REAL(KIND=dp), INTENT(IN)                          :: cutoff

      CHARACTER(len=*), PARAMETER                        :: routineN = 'pw_grid_assign'

      INTEGER                                            :: cnt, gpt, handle, i, ip, l, lby, lbz, ll, &
                                                            m, mm, n, n_upperlimit, nlo, nn, span
      INTEGER(KIND=int_8)                                :: gpt_global
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: plane_start
      INTEGER, DIMENSION(2, 3)                           :: bol, bounds
      LOGICAL                                            :: found_g0
      REAL(KIND=dp)                                      :: length, length_x, length_y, length_z

      CALL timeset(routineN, handle)

      bounds = pw_grid%bounds
      lby = pw_grid%bounds(1, 2)
      lbz = pw_grid%bounds(1, 3)

      IF (pw_grid%grid_span == HALFSPACE) THEN
         n_upperlimit = 0
      ELSE IF (pw_grid%grid_span == FULLSPACE) THEN
         n_upperlimit = bounds(2, 3)
      ELSE
         CPABORT("No type set for the grid")
      END IF

      ! finds valid g-points within grid
      IF (pw_grid%para%mode == PW_MODE_LOCAL) THEN
         ! The points of plane n follow those of the planes before it, in the order of the
         ! sequential loop: count every plane, then fill the planes in parallel.
         span = pw_grid%grid_span
         nlo = bounds(1, 3)
         ALLOCATE (plane_start(nlo:MAX(nlo, n_upperlimit) + 1))
         plane_start(:) = 0
!$OMP PARALLEL DEFAULT(NONE) SHARED(bounds, cutoff, h_inv, n_upperlimit, nlo, plane_start, pw_grid, span) &
!$OMP          PRIVATE(cnt, gpt, l, length, length_x, length_y, length_z, m, n)
!$OMP DO SCHEDULE(DYNAMIC, 1)
         DO n = nlo, n_upperlimit
            cnt = 0
            DO m = bounds(1, 2), bounds(2, 2)
               DO l = bounds(1, 1), bounds(2, 1)
                  IF (span == HALFSPACE .AND. n == 0) THEN
                     IF ((m == 0 .AND. l > 0) .OR. (m > 0)) CYCLE
                  END IF

                  CALL pw_vec_length(h_inv, length_x, length_y, length_z, length, l, m, n)

                  IF (0.5_dp*length <= cutoff) cnt = cnt + 1
               END DO
            END DO
            plane_start(n + 1) = cnt
         END DO
!$OMP END DO
!$OMP SINGLE
         DO n = nlo, n_upperlimit
            plane_start(n + 1) = plane_start(n + 1) + plane_start(n)
         END DO
!$OMP END SINGLE
!$OMP DO SCHEDULE(DYNAMIC, 1)
         DO n = nlo, n_upperlimit
            gpt = plane_start(n)
            DO m = bounds(1, 2), bounds(2, 2)
               DO l = bounds(1, 1), bounds(2, 1)
                  IF (span == HALFSPACE .AND. n == 0) THEN
                     IF ((m == 0 .AND. l > 0) .OR. (m > 0)) CYCLE
                  END IF

                  CALL pw_vec_length(h_inv, length_x, length_y, length_z, length, l, m, n)

                  IF (0.5_dp*length <= cutoff) THEN
                     gpt = gpt + 1
                     pw_grid%g(1, gpt) = length_x
                     pw_grid%g(2, gpt) = length_y
                     pw_grid%g(3, gpt) = length_z
                     pw_grid%gsq(gpt) = length
                     pw_grid%g_hat(1, gpt) = l
                     pw_grid%g_hat(2, gpt) = m
                     pw_grid%g_hat(3, gpt) = n
                  END IF

               END DO
            END DO
         END DO
!$OMP END DO
!$OMP END PARALLEL
         IF (n_upperlimit >= nlo) THEN
            gpt = plane_start(n_upperlimit + 1)
         ELSE
            gpt = 0
         END IF
         DEALLOCATE (plane_start)

      ELSE

         IF (pw_grid%para%ray_distribution) THEN

            gpt = 0
            ip = pw_grid%para%group%mepos
            DO i = 1, pw_grid%para%nyzray(ip)
               n = pw_grid%para%yzp(2, i, ip) + lbz - 1
               m = pw_grid%para%yzp(1, i, ip) + lby - 1
               IF (n > n_upperlimit) CYCLE
               DO l = bounds(1, 1), bounds(2, 1)
                  IF (pw_grid%grid_span == HALFSPACE .AND. n == 0) THEN
                     IF ((m == 0 .AND. l > 0) .OR. (m > 0)) CYCLE
                  END IF

                  CALL pw_vec_length(h_inv, length_x, length_y, length_z, length, l, m, n)

                  IF (0.5_dp*length <= cutoff) THEN
                     gpt = gpt + 1
                     pw_grid%g(1, gpt) = length_x
                     pw_grid%g(2, gpt) = length_y
                     pw_grid%g(3, gpt) = length_z
                     pw_grid%gsq(gpt) = length
                     pw_grid%g_hat(1, gpt) = l
                     pw_grid%g_hat(2, gpt) = m
                     pw_grid%g_hat(3, gpt) = n
                  END IF

               END DO
            END DO

         ELSE

            bol = pw_grid%para%bo(:, :, pw_grid%para%group%mepos, 3)
            gpt = 0
            DO n = bounds(1, 3), bounds(2, 3)
               IF (n < 0) THEN
                  nn = n + pw_grid%npts(3) + 1
               ELSE
                  nn = n + 1
               END IF
               IF (nn < bol(1, 3) .OR. nn > bol(2, 3)) CYCLE
               DO m = bounds(1, 2), bounds(2, 2)
                  IF (m < 0) THEN
                     mm = m + pw_grid%npts(2) + 1
                  ELSE
                     mm = m + 1
                  END IF
                  IF (mm < bol(1, 2) .OR. mm > bol(2, 2)) CYCLE
                  DO l = bounds(1, 1), bounds(2, 1)
                     IF (l < 0) THEN
                        ll = l + pw_grid%npts(1) + 1
                     ELSE
                        ll = l + 1
                     END IF
                     IF (ll < bol(1, 1) .OR. ll > bol(2, 1)) CYCLE

                     CALL pw_vec_length(h_inv, length_x, length_y, length_z, length, l, m, n)

                     gpt = gpt + 1
                     pw_grid%g(1, gpt) = length_x
                     pw_grid%g(2, gpt) = length_y
                     pw_grid%g(3, gpt) = length_z
                     pw_grid%gsq(gpt) = length
                     pw_grid%g_hat(1, gpt) = l
                     pw_grid%g_hat(2, gpt) = m
                     pw_grid%g_hat(3, gpt) = n

                  END DO
               END DO
            END DO

         END IF

      END IF

      ! Check the number of g-vectors for grid
      CPASSERT(pw_grid%ngpts_cut_local == gpt)
      IF (pw_grid%para%mode == PW_MODE_DISTRIBUTED) THEN
         gpt_global = gpt
         CALL pw_grid%para%group%sum(gpt_global)
         CPASSERT(pw_grid%ngpts_cut == gpt_global)
      END IF

      found_g0 = .FALSE.
!$OMP PARALLEL DO DEFAULT(NONE) SHARED(pw_grid) PRIVATE(gpt) REDUCTION(.OR.:found_g0)
      DO gpt = 1, pw_grid%ngpts_cut_local
         IF (pw_grid%g_hat(1, gpt) == 0 .AND. pw_grid%g_hat(2, gpt) == 0 .AND. &
             pw_grid%g_hat(3, gpt) == 0) found_g0 = .TRUE.
      END DO
!$OMP END PARALLEL DO
      pw_grid%have_g0 = found_g0
      pw_grid%first_gne0 = 1
      IF (found_g0) pw_grid%first_gne0 = 2

      CALL pw_grid_set_maps(pw_grid%grid_span, pw_grid%g_hat, &
                            pw_grid%mapl, pw_grid%mapm, pw_grid%mapn, pw_grid%npts)

      CALL timestop(handle)

   END SUBROUTINE pw_grid_assign

! **************************************************************************************************
!> \brief Setup maps from 1d to 3d space
!> \param grid_span ...
!> \param g_hat ...
!> \param mapl ...
!> \param mapm ...
!> \param mapn ...
!> \param npts ...
!> \par History
!>      JGH (21-12-2000) : Size of g_hat locally determined
!> \author apsi
!>      Christopher Mundy
!> \note
!>      Maps are to full 3D space (not distributed)
! **************************************************************************************************
   SUBROUTINE pw_grid_set_maps(grid_span, g_hat, mapl, mapm, mapn, npts)

      INTEGER, INTENT(IN)                                :: grid_span
      INTEGER, DIMENSION(:, :), INTENT(IN)               :: g_hat
      TYPE(map_pn), INTENT(INOUT)                        :: mapl, mapm, mapn
      INTEGER, DIMENSION(:), INTENT(IN)                  :: npts

      INTEGER                                            :: gpt, hi, lo, ng
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: my_seen, seen

      ng = SIZE(g_hat, 2)

      ! An entry of a map depends only on its own index; it is set for exactly the indices
      ! that occur in g_hat. Find those in parallel, then set the entries.
      lo = MIN(LBOUND(mapl%pos, 1), LBOUND(mapm%pos, 1), LBOUND(mapn%pos, 1))
      hi = MAX(UBOUND(mapl%pos, 1), UBOUND(mapm%pos, 1), UBOUND(mapn%pos, 1))
      ALLOCATE (seen(lo:hi, 3))
      seen(:, :) = 0
!$OMP PARALLEL DEFAULT(NONE) SHARED(g_hat, hi, lo, ng, seen) PRIVATE(gpt, my_seen)
      ALLOCATE (my_seen(lo:hi, 3))
      my_seen(:, :) = 0
!$OMP DO
      DO gpt = 1, ng
         my_seen(g_hat(1, gpt), 1) = 1
         my_seen(g_hat(2, gpt), 2) = 1
         my_seen(g_hat(3, gpt), 3) = 1
      END DO
!$OMP END DO NOWAIT
!$OMP CRITICAL(pw_grid_set_maps_seen)
      seen(:, :) = MAX(seen(:, :), my_seen(:, :))
!$OMP END CRITICAL(pw_grid_set_maps_seen)
      DEALLOCATE (my_seen)
!$OMP END PARALLEL

      CALL set_map_entries(mapl, seen(:, 1), lo, npts(1), grid_span)
      CALL set_map_entries(mapm, seen(:, 2), lo, npts(2), grid_span)
      CALL set_map_entries(mapn, seen(:, 3), lo, npts(3), grid_span)
      DEALLOCATE (seen)

   END SUBROUTINE pw_grid_set_maps

! **************************************************************************************************
!> \brief The map entries of the indices i with seen(i) /= 0, as in the loop over g_hat that
!>        pw_grid_set_maps replaces.
!> \param map ...
!> \param seen flags, seen(lo + k - 1) for index lo + k - 1
!> \param lo lower bound of seen
!> \param npts number of points along this direction
!> \param grid_span ...
! **************************************************************************************************
   SUBROUTINE set_map_entries(map, seen, lo, npts, grid_span)
      TYPE(map_pn), INTENT(INOUT)                        :: map
      INTEGER, DIMENSION(:), INTENT(IN)                  :: seen
      INTEGER, INTENT(IN)                                :: lo, npts, grid_span

      INTEGER                                            :: i

      DO i = LBOUND(map%pos, 1), UBOUND(map%pos, 1)
         IF (seen(i - lo + 1) == 0) CYCLE
         IF (i < 0) THEN
            map%pos(i) = i + npts
         ELSE
            map%pos(i) = i
         END IF
         ! Generating the maps to the full 3-d space
         IF (grid_span == HALFSPACE) THEN
            IF (i <= 0) THEN
               map%neg(i) = -i
            ELSE
               map%neg(i) = npts - i
            END IF
         END IF
      END DO

   END SUBROUTINE set_map_entries

! **************************************************************************************************
!> \brief Allocate all (Pointer) Arrays in pw_grid
!> \param pw_grid ...
!> \param ng ...
!> \param bounds ...
!> \par History
!>      JGH (20-12-2000) : Added status variable
!>                         Bounds of arrays now from calling routine, this
!>                         makes it independent from parallel setup
!> \author apsi
!>      Christopher Mundy
! **************************************************************************************************
   SUBROUTINE pw_grid_allocate(pw_grid, ng, bounds)

      ! Argument
      TYPE(pw_grid_type), INTENT(INOUT)        :: pw_grid
      INTEGER, INTENT(IN)                      :: ng
      INTEGER, DIMENSION(:, :), INTENT(IN)     :: bounds

      CHARACTER(len=*), PARAMETER :: routineN = 'pw_grid_allocate'

      INTEGER                                    :: nmaps
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      INTEGER(KIND=C_SIZE_T)                     :: length
      TYPE(C_PTR)                                :: cptr_g_hatmap
      INTEGER                                    :: stat
#endif

      INTEGER                                  :: handle

      CALL timeset(routineN, handle)

      ALLOCATE (pw_grid%g(3, ng))
      ALLOCATE (pw_grid%gsq(ng))
      ALLOCATE (pw_grid%g_hat(3, ng))

      nmaps = 1
      IF (pw_grid%grid_span == HALFSPACE) nmaps = 2
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      CALL offload_activate_chosen_device()

      length = INT(int_size*MAX(ng, 1)*MAX(nmaps, 1), KIND=C_SIZE_T)
      stat = offload_malloc_pinned_mem(cptr_g_hatmap, length)
      CPASSERT(stat == 0)
      CALL c_f_pointer(cptr_g_hatmap, pw_grid%g_hatmap, [MAX(ng, 1), MAX(nmaps, 1)])
#else
      ALLOCATE (pw_grid%g_hatmap(1, 1))
#endif

      IF (pw_grid%para%mode == PW_MODE_DISTRIBUTED) THEN
         ALLOCATE (pw_grid%grays(pw_grid%npts(1), &
                                 pw_grid%para%nyzray(pw_grid%para%group%mepos)))
      END IF

      ALLOCATE (pw_grid%mapl%pos(bounds(1, 1):bounds(2, 1)))
      ALLOCATE (pw_grid%mapl%neg(bounds(1, 1):bounds(2, 1)))
      ALLOCATE (pw_grid%mapm%pos(bounds(1, 2):bounds(2, 2)))
      ALLOCATE (pw_grid%mapm%neg(bounds(1, 2):bounds(2, 2)))
      ALLOCATE (pw_grid%mapn%pos(bounds(1, 3):bounds(2, 3)))
      ALLOCATE (pw_grid%mapn%neg(bounds(1, 3):bounds(2, 3)))

      CALL timestop(handle)

   END SUBROUTINE pw_grid_allocate

! **************************************************************************************************
!> \brief Sort g-vectors according to length
!> \param pw_grid ...
!> \param ref_grid ...
!> \par History
!>      JGH (20-12-2000) : allocate idx, ng = SIZE ( pw_grid % gsq ) the
!>                         sorting is local and independent from parallelisation
!>                         WARNING: Global ordering depends now on the number
!>                                  of cpus.
!>      JGH (28-02-2001) : check for ordering against reference grid
!>      JGH (01-05-2001) : sort spherical cutoff grids also within shells
!>                         reference grids for non-spherical cutoffs
!>      JGH (20-06-2001) : do not sort non-spherical grids
!>      JGH (19-02-2003) : Order all grids, this makes subgrids also for
!>                         non-spherical cutoffs possible
!>      JGH (21-02-2003) : Introduce gather array for general reference grids
!> \author apsi
!>      Christopher Mundy
! **************************************************************************************************
   SUBROUTINE pw_grid_sort(pw_grid, ref_grid)

      ! Argument
      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid
      TYPE(pw_grid_type), INTENT(IN), OPTIONAL           :: ref_grid

      CHARACTER(len=*), PARAMETER                        :: routineN = 'pw_grid_sort'

      INTEGER                                            :: handle, i, ig, ih, ip, is, it, j, ng, ngr
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: idx
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: int_tmp
      LOGICAL                                            :: g_found
      REAL(KIND=dp)                                      :: gig, gigr
      REAL(KIND=dp), CONTIGUOUS, DIMENSION(:, :), &
         POINTER                                         :: real_tmp

      CALL timeset(routineN, handle)

      ng = SIZE(pw_grid%gsq)
      ALLOCATE (idx(ng))

      ! grids are (locally) ordered by length of G-vectors
      ! (a stable sort: the result is the same as that of util's sort, see sort_gsq)
      CALL sort_gsq(pw_grid%gsq, ng, idx)
      ! within shells order wrt x,y,z
      CALL sort_shells(pw_grid%gsq, pw_grid%g_hat, idx)

      ! Gather g and g_hat in the sorted order into new arrays that replace the old ones
      ! (no copy back). g has the same shape (3, ng) as allocated in pw_grid_allocate.
      ALLOCATE (real_tmp(3, ng), int_tmp(3, ng))
!$OMP PARALLEL DO DEFAULT(NONE) SHARED(idx, int_tmp, ng, pw_grid, real_tmp) PRIVATE(i, j)
      DO i = 1, ng
         j = idx(i)
         real_tmp(1, i) = pw_grid%g(1, j)
         real_tmp(2, i) = pw_grid%g(2, j)
         real_tmp(3, i) = pw_grid%g(3, j)
         int_tmp(1, i) = pw_grid%g_hat(1, j)
         int_tmp(2, i) = pw_grid%g_hat(2, j)
         int_tmp(3, i) = pw_grid%g_hat(3, j)
      END DO
!$OMP END PARALLEL DO
      DEALLOCATE (pw_grid%g)
      pw_grid%g => real_tmp
      NULLIFY (real_tmp)
      CALL MOVE_ALLOC(int_tmp, pw_grid%g_hat)

      DEALLOCATE (idx)

      ! check if ordering is compatible to reference grid
      IF (PRESENT(ref_grid)) THEN
         ngr = SIZE(ref_grid%gsq)
         ngr = MIN(ng, ngr)
         IF (pw_grid%spherical) THEN
            IF (.NOT. ALL(pw_grid%g_hat(1:3, 1:ngr) &
                          == ref_grid%g_hat(1:3, 1:ngr))) THEN
               CPABORT("G space sorting not compatible")
            END IF
         ELSE
            ALLOCATE (pw_grid%gidx(1:ngr))
            ! first try as many trivial associations as possible: the G-vectors before the
            ! first one that differs from the reference grid (it = that position - 1)
            it = ngr
!$OMP PARALLEL DO DEFAULT(NONE) SHARED(ngr, pw_grid, ref_grid) PRIVATE(ig) REDUCTION(MIN:it)
            DO ig = 1, ngr
               IF (pw_grid%g_hat(1, ig) /= ref_grid%g_hat(1, ig) .OR. &
                   pw_grid%g_hat(2, ig) /= ref_grid%g_hat(2, ig) .OR. &
                   pw_grid%g_hat(3, ig) /= ref_grid%g_hat(3, ig)) it = MIN(it, ig - 1)
            END DO
!$OMP END PARALLEL DO
!$OMP PARALLEL DO DEFAULT(NONE) SHARED(it, ngr, pw_grid) PRIVATE(ig)
            DO ig = 1, ngr
               IF (ig <= it) THEN
                  pw_grid%gidx(ig) = ig
               ELSE
                  pw_grid%gidx(ig) = 0
               END IF
            END DO
!$OMP END PARALLEL DO
            ! now for the ones that could not be done
            IF (ng == ngr .AND. use_gidx_table(pw_grid, ref_grid)) THEN
               ! for the case pw_grid <= ref_grid: direct lookup of the (unique) Miller indices
               CALL gidx_from_table(pw_grid, ref_grid, it, ngr)
            ELSE IF (ng == ngr) THEN
               ! for the case pw_grid <= ref_grid
               is = it
               DO ig = it + 1, ngr
                  gig = pw_grid%gsq(ig)
                  gigr = MAX(1._dp, gig)
                  g_found = .FALSE.
                  DO ih = is + 1, SIZE(ref_grid%gsq)
                     IF (ABS(gig - ref_grid%gsq(ih))/gigr > 1.e-12_dp) CYCLE
                     g_found = .TRUE.
                     EXIT
                  END DO
                  IF (.NOT. g_found) THEN
                     WRITE (*, "(A,I10,F20.10)") "G-vector", ig, pw_grid%gsq(ig)
                     CPABORT("G vector not found")
                  END IF
                  ip = ih - 1
                  DO ih = ip + 1, SIZE(ref_grid%gsq)
                     IF (ABS(gig - ref_grid%gsq(ih))/gigr > 1.e-12_dp) CYCLE
                     IF (pw_grid%g_hat(1, ig) /= ref_grid%g_hat(1, ih)) CYCLE
                     IF (pw_grid%g_hat(2, ig) /= ref_grid%g_hat(2, ih)) CYCLE
                     IF (pw_grid%g_hat(3, ig) /= ref_grid%g_hat(3, ih)) CYCLE
                     pw_grid%gidx(ig) = ih
                     EXIT
                  END DO
                  IF (pw_grid%gidx(ig) == 0) THEN
                     WRITE (*, "(A,2I10)") " G-Shell ", is + 1, ip + 1
                     WRITE (*, "(A,I10,3I6,F20.10)") &
                        " G-vector", ig, pw_grid%g_hat(1:3, ig), pw_grid%gsq(ig)
                     DO ih = 1, SIZE(ref_grid%gsq)
                        IF (pw_grid%g_hat(1, ig) /= ref_grid%g_hat(1, ih)) CYCLE
                        IF (pw_grid%g_hat(2, ig) /= ref_grid%g_hat(2, ih)) CYCLE
                        IF (pw_grid%g_hat(3, ig) /= ref_grid%g_hat(3, ih)) CYCLE
                        WRITE (*, "(A,I10,3I6,F20.10)") &
                           " G-vector", ih, ref_grid%g_hat(1:3, ih), ref_grid%gsq(ih)
                     END DO
                     CPABORT("G vector not found")
                  END IF
                  is = ip
               END DO
            ELSE
               ! for the case pw_grid > ref_grid
               is = it
               DO ig = it + 1, ngr
                  gig = ref_grid%gsq(ig)
                  gigr = MAX(1._dp, gig)
                  g_found = .FALSE.
                  DO ih = is + 1, ng
                     IF (ABS(pw_grid%gsq(ih) - gig)/gigr > 1.e-12_dp) CYCLE
                     g_found = .TRUE.
                     EXIT
                  END DO
                  IF (.NOT. g_found) THEN
                     WRITE (*, "(A,I10,F20.10)") "G-vector", ig, ref_grid%gsq(ig)
                     CPABORT("G vector not found")
                  END IF
                  ip = ih - 1
                  DO ih = ip + 1, ng
                     IF (ABS(pw_grid%gsq(ih) - gig)/gigr > 1.e-12_dp) CYCLE
                     IF (pw_grid%g_hat(1, ih) /= ref_grid%g_hat(1, ig)) CYCLE
                     IF (pw_grid%g_hat(2, ih) /= ref_grid%g_hat(2, ig)) CYCLE
                     IF (pw_grid%g_hat(3, ih) /= ref_grid%g_hat(3, ig)) CYCLE
                     pw_grid%gidx(ig) = ih
                     EXIT
                  END DO
                  IF (pw_grid%gidx(ig) == 0) THEN
                     WRITE (*, "(A,2I10)") " G-Shell ", is + 1, ip + 1
                     WRITE (*, "(A,I10,3I6,F20.10)") &
                        " G-vector", ig, ref_grid%g_hat(1:3, ig), ref_grid%gsq(ig)
                     DO ih = 1, ng
                        IF (pw_grid%g_hat(1, ih) /= ref_grid%g_hat(1, ig)) CYCLE
                        IF (pw_grid%g_hat(2, ih) /= ref_grid%g_hat(2, ig)) CYCLE
                        IF (pw_grid%g_hat(3, ih) /= ref_grid%g_hat(3, ig)) CYCLE
                        WRITE (*, "(A,I10,3I6,F20.10)") &
                           " G-vector", ih, pw_grid%g_hat(1:3, ih), pw_grid%gsq(ih)
                     END DO
                     CPABORT("G vector not found")
                  END IF
                  is = ip
               END DO
            END IF
            ! test if all g-vectors are associated
            IF (ANY(pw_grid%gidx == 0)) THEN
               CPABORT("G space sorting not compatible")
            END IF
         END IF
      END IF

      !check if G=0 is at first position
      IF (pw_grid%have_g0) THEN
         IF (pw_grid%gsq(1) /= 0.0_dp) THEN
            CPABORT("G = 0 not in first position")
         END IF
      END IF

      CALL timestop(handle)

   END SUBROUTINE pw_grid_sort

! **************************************************************************************************
!> \brief Stable ascending sort of arr, returning the original positions in idx. Chunks are sorted
!>        concurrently with util's (stable merge) sort and merged pairwise, taking the right run
!>        only if it is strictly smaller. Ties therefore keep their original order and the result
!>        (arr and idx) is identical to CALL sort(arr, n, idx).
!> \param arr ...
!> \param n ...
!> \param idx ...
! **************************************************************************************************
   SUBROUTINE sort_gsq(arr, n, idx)
      INTEGER, INTENT(IN)                                :: n
      REAL(KIND=dp), DIMENSION(1:n), INTENT(INOUT)       :: arr
      INTEGER, DIMENSION(1:n), INTENT(INOUT)             :: idx

      CHARACTER(len=*), PARAMETER                        :: routineN = 'sort_gsq'

      INTEGER                                            :: c, handle, hi, i, lo, mid, nchunk, &
                                                            nthreads, width
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: bounds, tmp_idx
      LOGICAL                                            :: in_tmp
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: tmp_arr

      nthreads = 1
!$    nthreads = omp_get_max_threads()
!$    IF (omp_in_parallel()) nthreads = 1
      IF (nthreads < 2 .OR. n < 65536) THEN
         CALL sort(arr, n, idx)
         RETURN
      END IF
      CALL timeset(routineN, handle)

      nchunk = 1
      DO WHILE (nchunk < nthreads .AND. nchunk < 256)
         nchunk = 2*nchunk
      END DO
      ALLOCATE (bounds(0:nchunk))
      DO c = 0, nchunk
         bounds(c) = INT(INT(c, int_8)*INT(n, int_8)/INT(nchunk, int_8))
      END DO

!$OMP PARALLEL DO DEFAULT(NONE) SHARED(arr, bounds, idx, nchunk) PRIVATE(c, hi, i, lo) &
!$OMP             SCHEDULE(DYNAMIC, 1)
      DO c = 1, nchunk
         lo = bounds(c - 1) + 1
         hi = bounds(c)
         CALL sort(arr(lo:hi), hi - lo + 1, idx(lo:hi))
         DO i = lo, hi
            idx(i) = idx(i) + lo - 1
         END DO
      END DO
!$OMP END PARALLEL DO

      ALLOCATE (tmp_arr(n), tmp_idx(n))
      in_tmp = .FALSE.
      width = 1
      DO WHILE (width < nchunk)
!$OMP PARALLEL DO DEFAULT(NONE) SHARED(arr, bounds, idx, in_tmp, nchunk, tmp_arr, tmp_idx, width) &
!$OMP             PRIVATE(c, hi, lo, mid) SCHEDULE(DYNAMIC, 1)
         DO c = 0, nchunk - 1, 2*width
            lo = bounds(c) + 1
            mid = bounds(MIN(c + width, nchunk))
            hi = bounds(MIN(c + 2*width, nchunk))
            IF (in_tmp) THEN
               CALL merge_runs(tmp_arr, tmp_idx, arr, idx, lo, mid, hi)
            ELSE
               CALL merge_runs(arr, idx, tmp_arr, tmp_idx, lo, mid, hi)
            END IF
         END DO
!$OMP END PARALLEL DO
         in_tmp = .NOT. in_tmp
         width = 2*width
      END DO

      IF (in_tmp) THEN
!$OMP PARALLEL DO DEFAULT(NONE) SHARED(arr, idx, n, tmp_arr, tmp_idx) PRIVATE(i)
         DO i = 1, n
            arr(i) = tmp_arr(i)
            idx(i) = tmp_idx(i)
         END DO
!$OMP END PARALLEL DO
      END IF
      DEALLOCATE (bounds, tmp_arr, tmp_idx)
      CALL timestop(handle)

   END SUBROUTINE sort_gsq

! **************************************************************************************************
!> \brief Stable merge of the sorted runs src(lo:mid) and src(mid+1:hi) into dst(lo:hi).
!> \param src_a ...
!> \param src_i ...
!> \param dst_a ...
!> \param dst_i ...
!> \param lo ...
!> \param mid ...
!> \param hi ...
! **************************************************************************************************
   SUBROUTINE merge_runs(src_a, src_i, dst_a, dst_i, lo, mid, hi)
      REAL(KIND=dp), DIMENSION(:), INTENT(IN)            :: src_a
      INTEGER, DIMENSION(:), INTENT(IN)                  :: src_i
      REAL(KIND=dp), DIMENSION(:), INTENT(INOUT)         :: dst_a
      INTEGER, DIMENSION(:), INTENT(INOUT)               :: dst_i
      INTEGER, INTENT(IN)                                :: lo, mid, hi

      INTEGER                                            :: i, j, k

      i = lo
      j = mid + 1
      k = lo
      DO WHILE (i <= mid .AND. j <= hi)
         IF (src_a(j) < src_a(i)) THEN
            dst_a(k) = src_a(j)
            dst_i(k) = src_i(j)
            j = j + 1
         ELSE
            dst_a(k) = src_a(i)
            dst_i(k) = src_i(i)
            i = i + 1
         END IF
         k = k + 1
      END DO
      DO WHILE (i <= mid)
         dst_a(k) = src_a(i)
         dst_i(k) = src_i(i)
         i = i + 1
         k = k + 1
      END DO
      DO WHILE (j <= hi)
         dst_a(k) = src_a(j)
         dst_i(k) = src_i(j)
         j = j + 1
         k = k + 1
      END DO

   END SUBROUTINE merge_runs

! **************************************************************************************************
!> \brief Whether gidx_from_table applies: every reference Miller index lies within the bounds
!>        of the reference grid and the lookup table stays within a modest size.
!> \param pw_grid ...
!> \param ref_grid ...
!> \return ...
! **************************************************************************************************
   FUNCTION use_gidx_table(pw_grid, ref_grid) RESULT(ok)
      TYPE(pw_grid_type), INTENT(IN)                     :: pw_grid, ref_grid
      LOGICAL                                            :: ok

      INTEGER                                            :: d, hi1, hi2, hi3, ih, lo1, lo2, lo3, &
                                                            nref
      INTEGER(KIND=int_8)                                :: npts

      ok = ALLOCATED(ref_grid%g_hat) .AND. ASSOCIATED(ref_grid%gsq) .AND. &
           ALLOCATED(pw_grid%g_hat) .AND. ASSOCIATED(pw_grid%gidx)
      IF (.NOT. ok) RETURN
      npts = 1
      DO d = 1, 3
         ok = ok .AND. ref_grid%bounds(2, d) >= ref_grid%bounds(1, d)
         IF (.NOT. ok) RETURN
         npts = npts*INT(ref_grid%bounds(2, d) - ref_grid%bounds(1, d) + 1, int_8)
      END DO
      ! 4 bytes per point; the reference grids of the multigrids stay well below this limit
      ok = npts <= 2_int_8**30
      IF (.NOT. ok) RETURN
      ok = SIZE(ref_grid%g_hat, 2) >= SIZE(ref_grid%gsq) .AND. SIZE(ref_grid%g_hat, 1) == 3 .AND. &
           SIZE(pw_grid%g_hat, 1) == 3
      IF (.NOT. ok) RETURN
      ! MINVAL/MAXVAL of g_hat(d, 1:nref) for d = 1, 2, 3 in one parallel pass
      nref = SIZE(ref_grid%gsq)
      lo1 = HUGE(0)
      lo2 = HUGE(0)
      lo3 = HUGE(0)
      hi1 = -HUGE(0)
      hi2 = -HUGE(0)
      hi3 = -HUGE(0)
!$OMP PARALLEL DO DEFAULT(NONE) SHARED(nref, ref_grid) PRIVATE(ih) &
!$OMP             REDUCTION(MIN:lo1, lo2, lo3) REDUCTION(MAX:hi1, hi2, hi3)
      DO ih = 1, nref
         lo1 = MIN(lo1, ref_grid%g_hat(1, ih))
         lo2 = MIN(lo2, ref_grid%g_hat(2, ih))
         lo3 = MIN(lo3, ref_grid%g_hat(3, ih))
         hi1 = MAX(hi1, ref_grid%g_hat(1, ih))
         hi2 = MAX(hi2, ref_grid%g_hat(2, ih))
         hi3 = MAX(hi3, ref_grid%g_hat(3, ih))
      END DO
!$OMP END PARALLEL DO
      ok = lo1 >= ref_grid%bounds(1, 1) .AND. hi1 <= ref_grid%bounds(2, 1) .AND. &
           lo2 >= ref_grid%bounds(1, 2) .AND. hi2 <= ref_grid%bounds(2, 2) .AND. &
           lo3 >= ref_grid%bounds(1, 3) .AND. hi3 <= ref_grid%bounds(2, 3)

   END FUNCTION use_gidx_table

! **************************************************************************************************
!> \brief gidx for pw_grid <= ref_grid. The Miller indices of a grid are unique, so the reference
!>        position of every G is found by a direct lookup. It is accepted under the same condition
!>        as in the sequential search of pw_grid_sort (|G|^2 agrees to 1.e-12 relative), which
!>        makes the result identical to that search.
!> \param pw_grid ...
!> \param ref_grid ...
!> \param it number of leading G-vectors that coincide with the reference grid
!> \param ngr ...
! **************************************************************************************************
   SUBROUTINE gidx_from_table(pw_grid, ref_grid, it, ngr)
      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid
      TYPE(pw_grid_type), INTENT(IN)                     :: ref_grid
      INTEGER, INTENT(IN)                                :: it, ngr

      CHARACTER(len=*), PARAMETER                        :: routineN = 'gidx_from_table'

      INTEGER                                            :: bad, handle, hi1, hi2, hi3, ig, ih, l, &
                                                            lo1, lo2, lo3, m, n
      INTEGER, ALLOCATABLE, DIMENSION(:, :, :)           :: table
      INTEGER, DIMENSION(2, 3)                           :: rb
      REAL(KIND=dp)                                      :: gig, gigr

      IF (it >= ngr) RETURN
      CALL timeset(routineN, handle)

      ! Only the Miller indices of G-vectors it+1:ngr are looked up: the table covers the box
      ! they span within the reference bounds, filled from the reference G-vectors in it.
      lo1 = HUGE(0)
      lo2 = HUGE(0)
      lo3 = HUGE(0)
      hi1 = -HUGE(0)
      hi2 = -HUGE(0)
      hi3 = -HUGE(0)
!$OMP PARALLEL DO DEFAULT(NONE) SHARED(it, ngr, pw_grid) PRIVATE(ig) &
!$OMP             REDUCTION(MIN:lo1, lo2, lo3) REDUCTION(MAX:hi1, hi2, hi3)
      DO ig = it + 1, ngr
         lo1 = MIN(lo1, pw_grid%g_hat(1, ig))
         lo2 = MIN(lo2, pw_grid%g_hat(2, ig))
         lo3 = MIN(lo3, pw_grid%g_hat(3, ig))
         hi1 = MAX(hi1, pw_grid%g_hat(1, ig))
         hi2 = MAX(hi2, pw_grid%g_hat(2, ig))
         hi3 = MAX(hi3, pw_grid%g_hat(3, ig))
      END DO
!$OMP END PARALLEL DO
      rb(1, 1) = MAX(lo1, ref_grid%bounds(1, 1))
      rb(1, 2) = MAX(lo2, ref_grid%bounds(1, 2))
      rb(1, 3) = MAX(lo3, ref_grid%bounds(1, 3))
      rb(2, 1) = MIN(hi1, ref_grid%bounds(2, 1))
      rb(2, 2) = MIN(hi2, ref_grid%bounds(2, 2))
      rb(2, 3) = MIN(hi3, ref_grid%bounds(2, 3))
      ALLOCATE (table(rb(1, 1):rb(2, 1), rb(1, 2):rb(2, 2), rb(1, 3):rb(2, 3)))
!$OMP PARALLEL DO DEFAULT(NONE) SHARED(rb, table) PRIVATE(n)
      DO n = rb(1, 3), rb(2, 3)
         table(:, :, n) = 0
      END DO
!$OMP END PARALLEL DO
!$OMP PARALLEL DO DEFAULT(NONE) SHARED(rb, ref_grid, table) PRIVATE(ih, l, m, n)
      DO ih = 1, SIZE(ref_grid%gsq)
         l = ref_grid%g_hat(1, ih)
         m = ref_grid%g_hat(2, ih)
         n = ref_grid%g_hat(3, ih)
         IF (l >= rb(1, 1) .AND. l <= rb(2, 1) .AND. m >= rb(1, 2) .AND. m <= rb(2, 2) .AND. &
             n >= rb(1, 3) .AND. n <= rb(2, 3)) table(l, m, n) = ih
      END DO
!$OMP END PARALLEL DO

      bad = HUGE(0)
!$OMP PARALLEL DO DEFAULT(NONE) SHARED(it, ngr, pw_grid, rb, ref_grid, table) &
!$OMP             PRIVATE(gig, gigr, ig, ih, l, m, n) REDUCTION(MIN:bad)
      DO ig = it + 1, ngr
         l = pw_grid%g_hat(1, ig)
         m = pw_grid%g_hat(2, ig)
         n = pw_grid%g_hat(3, ig)
         ih = 0
         IF (l >= rb(1, 1) .AND. l <= rb(2, 1) .AND. m >= rb(1, 2) .AND. m <= rb(2, 2) .AND. &
             n >= rb(1, 3) .AND. n <= rb(2, 3)) ih = table(l, m, n)
         IF (ih > 0) THEN
            gig = pw_grid%gsq(ig)
            gigr = MAX(1._dp, gig)
            IF (ABS(gig - ref_grid%gsq(ih))/gigr > 1.e-12_dp) ih = 0
         END IF
         pw_grid%gidx(ig) = ih
         IF (ih == 0) bad = MIN(bad, ig)
      END DO
!$OMP END PARALLEL DO
      DEALLOCATE (table)

      IF (bad /= HUGE(0)) THEN
         WRITE (*, "(A,I10,3I6,F20.10)") &
            " G-vector", bad, pw_grid%g_hat(1:3, bad), pw_grid%gsq(bad)
         CPABORT("G vector not found")
      END IF
      CALL timestop(handle)

   END SUBROUTINE gidx_from_table

! **************************************************************************************************
!> \brief ...
!> \param gsq ...
!> \param g_hat ...
!> \param idx ...
! **************************************************************************************************
   SUBROUTINE sort_shells(gsq, g_hat, idx)

      ! Argument
      REAL(KIND=dp), DIMENSION(:), INTENT(IN)            :: gsq
      INTEGER, DIMENSION(:, :), INTENT(IN)               :: g_hat
      INTEGER, DIMENSION(:), INTENT(INOUT)               :: idx

      CHARACTER(len=*), PARAMETER                        :: routineN = 'sort_shells'
      REAL(KIND=dp), PARAMETER                           :: small = 5.e-16_dp

      INTEGER                                            :: b, handle, ig, ish, k, nblk, ng, nshell
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: seg_first, seg_shells, shell_start
      REAL(KIND=dp)                                      :: s_begin

      CALL timeset(routineN, handle)

! Juergs temporary hack to get the grids sorted for large (4000Ry) cutoffs.
! might need to call lapack for machine precision.

      ! A shell starts where gsq leaves the first value of the current shell by at least small.
      ! gsq is sorted, so a step gsq(ig) - gsq(ig-1) >= small certainly starts a shell, and the
      ! sequential search restarts there in the same state. Segments that begin at such steps
      ! are therefore searched independently; each shell is then reordered independently.
      ng = SIZE(gsq)
      nblk = 1
!$    nblk = omp_get_max_threads()
!$    IF (omp_in_parallel()) nblk = 1
      IF (ng < 65536) nblk = 1
      ALLOCATE (seg_first(0:nblk), seg_shells(0:nblk))
!$OMP PARALLEL DO DEFAULT(NONE) SHARED(gsq, nblk, ng, seg_first) PRIVATE(b, ig)
      DO b = 0, nblk - 1
         ig = INT(INT(b, int_8)*INT(ng, int_8)/INT(nblk, int_8)) + 1
         IF (b > 0) THEN
            DO WHILE (ig <= ng)
               IF (.NOT. (gsq(ig) - gsq(ig - 1) < small)) EXIT
               ig = ig + 1
            END DO
         END IF
         seg_first(b) = ig
      END DO
!$OMP END PARALLEL DO
      seg_first(nblk) = ng + 1

!$OMP PARALLEL DO DEFAULT(NONE) SHARED(gsq, nblk, seg_first, seg_shells) PRIVATE(b, ig, k, s_begin) &
!$OMP             SCHEDULE(DYNAMIC, 1)
      DO b = 0, nblk - 1
         k = 0
         s_begin = -1.0_dp
         DO ig = seg_first(b), seg_first(b + 1) - 1
            IF (.NOT. (ABS(gsq(ig) - s_begin) < small)) THEN
               k = k + 1
               s_begin = gsq(ig)
            END IF
         END DO
         seg_shells(b) = k
      END DO
!$OMP END PARALLEL DO
      nshell = 0
      DO b = 0, nblk - 1
         k = seg_shells(b)
         seg_shells(b) = nshell
         nshell = nshell + k
      END DO

      ALLOCATE (shell_start(nshell + 1))
!$OMP PARALLEL DO DEFAULT(NONE) SHARED(gsq, nblk, seg_first, seg_shells, shell_start) &
!$OMP             PRIVATE(b, ig, k, s_begin) SCHEDULE(DYNAMIC, 1)
      DO b = 0, nblk - 1
         k = seg_shells(b)
         s_begin = -1.0_dp
         DO ig = seg_first(b), seg_first(b + 1) - 1
            IF (.NOT. (ABS(gsq(ig) - s_begin) < small)) THEN
               k = k + 1
               shell_start(k) = ig
               s_begin = gsq(ig)
            END IF
         END DO
      END DO
!$OMP END PARALLEL DO
      shell_start(nshell + 1) = ng + 1
      DEALLOCATE (seg_first, seg_shells)

!$OMP PARALLEL DO DEFAULT(NONE) SHARED(g_hat, idx, nshell, shell_start) PRIVATE(ish) &
!$OMP             SCHEDULE(DYNAMIC, 4096)
      DO ish = 1, nshell
         IF (shell_start(ish + 1) - shell_start(ish) > 1) &
            CALL redist(g_hat, idx, shell_start(ish), shell_start(ish + 1) - 1)
      END DO
!$OMP END PARALLEL DO
      DEALLOCATE (shell_start)

      CALL timestop(handle)

   END SUBROUTINE sort_shells

! **************************************************************************************************
!> \brief ...
!> \param g_hat ...
!> \param idx ...
!> \param s1 ...
!> \param s2 ...
! **************************************************************************************************
   SUBROUTINE redist(g_hat, idx, s1, s2)

      ! Argument
      INTEGER, DIMENSION(:, :), INTENT(IN)               :: g_hat
      INTEGER, DIMENSION(:), INTENT(INOUT)               :: idx
      INTEGER, INTENT(IN)                                :: s1, s2

      INTEGER, PARAMETER                                 :: nsmall = 64

      INTEGER                                            :: i, ii, j, n1, n2, n3, ns, v
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: indl
      INTEGER, DIMENSION(nsmall)                         :: sval
      REAL(KIND=dp)                                      :: k
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: slen
      REAL(KIND=dp), DIMENSION(nsmall)                   :: skey

      IF (s2 <= s1) RETURN
      ns = s2 - s1 + 1
      IF (ns <= nsmall) THEN
         ! Same keys and the same (stable) order as util's sort below, without heap scratch
         DO i = 1, ns
            ii = idx(s1 + i - 1)
            n1 = g_hat(1, ii)
            n2 = g_hat(2, ii)
            n3 = g_hat(3, ii)
            skey(i) = 1000.0_dp*REAL(n1, dp) + &
                      REAL(n2, dp) + 0.001_dp*REAL(n3, dp)
            sval(i) = ii
         END DO
         DO i = 2, ns
            k = skey(i)
            v = sval(i)
            j = i - 1
            DO WHILE (j >= 1)
               IF (.NOT. (k < skey(j))) EXIT
               skey(j + 1) = skey(j)
               sval(j + 1) = sval(j)
               j = j - 1
            END DO
            skey(j + 1) = k
            sval(j + 1) = v
         END DO
         idx(s1:s2) = sval(1:ns)
         RETURN
      END IF
      ALLOCATE (indl(ns))
      ALLOCATE (slen(ns))

      DO i = s1, s2
         ii = idx(i)
         n1 = g_hat(1, ii)
         n2 = g_hat(2, ii)
         n3 = g_hat(3, ii)
         slen(i - s1 + 1) = 1000.0_dp*REAL(n1, dp) + &
                            REAL(n2, dp) + 0.001_dp*REAL(n3, dp)
      END DO
      CALL sort(slen, ns, indl)
      DO i = 1, ns
         ii = indl(i) + s1 - 1
         indl(i) = idx(ii)
      END DO
      idx(s1:s2) = indl(1:ns)

      DEALLOCATE (indl)
      DEALLOCATE (slen)

   END SUBROUTINE redist

! **************************************************************************************************
!> \brief Reorder yzq and yzp arrays for parallel FFT according to FFT mapping
!> \param pw_grid ...
!> \param yz ...
!> \par History
!>      none
!> \author JGH (17-Jan-2001)
! **************************************************************************************************
   SUBROUTINE pw_grid_remap(pw_grid, yz)

      ! Argument
      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid
      INTEGER, DIMENSION(:, :), INTENT(OUT)              :: yz

      CHARACTER(len=*), PARAMETER                        :: routineN = 'pw_grid_remap'

      INTEGER                                            :: gpt, handle, i, ip, is, j, m, n

      IF (pw_grid%para%mode == PW_MODE_LOCAL) RETURN

      CALL timeset(routineN, handle)

      ASSOCIATE (ny => pw_grid%npts(2), nz => pw_grid%npts(3), posm => pw_grid%mapm%pos, posn => pw_grid%mapn%pos, &
                 negm => pw_grid%mapm%neg, negn => pw_grid%mapn%neg)

         yz = 0
         pw_grid%para%yzp = 0
         pw_grid%para%yzq = 0

         DO gpt = 1, SIZE(pw_grid%gsq)
            m = posm(pw_grid%g_hat(2, gpt)) + 1
            n = posn(pw_grid%g_hat(3, gpt)) + 1
            yz(m, n) = yz(m, n) + 1
         END DO
         IF (pw_grid%grid_span == HALFSPACE) THEN
            DO gpt = 1, SIZE(pw_grid%gsq)
               m = negm(pw_grid%g_hat(2, gpt)) + 1
               n = negn(pw_grid%g_hat(3, gpt)) + 1
               yz(m, n) = yz(m, n) + 1
            END DO
         END IF

         ip = pw_grid%para%group%mepos
         is = 0
         DO i = 1, nz
            DO j = 1, ny
               IF (yz(j, i) > 0) THEN
                  is = is + 1
                  pw_grid%para%yzp(1, is, ip) = j
                  pw_grid%para%yzp(2, is, ip) = i
                  pw_grid%para%yzq(j, i) = is
               END IF
            END DO
         END DO
      END ASSOCIATE

      CPASSERT(is == pw_grid%para%nyzray(ip))
      CALL pw_grid%para%group%sum(pw_grid%para%yzp)

      CALL timestop(handle)

   END SUBROUTINE pw_grid_remap

! **************************************************************************************************
!> \brief Drops the device-side data (g_hatmap, g-vectors) that pw_gpu caches for a grid.
!> \param grid_id id_nr of the grid
! **************************************************************************************************
   SUBROUTINE pw_grid_gpu_invalidate(grid_id)
      INTEGER, INTENT(IN)                                :: grid_id

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      INTERFACE
         SUBROUTINE pw_gpu_grid_invalidate_c(id) BIND(C, name="pw_gpu_grid_invalidate")
            USE ISO_C_BINDING, ONLY: C_INT
            INTEGER(KIND=C_INT), INTENT(IN), VALUE       :: id
         END SUBROUTINE pw_gpu_grid_invalidate_c
      END INTERFACE

      CALL pw_gpu_grid_invalidate_c(grid_id)
#else
      MARK_USED(grid_id)
#endif
   END SUBROUTINE pw_grid_gpu_invalidate

! **************************************************************************************************
!> \brief Recalculate the g-vectors after a change of the box
!> \param cell_hmat ...
!> \param pw_grid ...
!> \par History
!>      JGH (20-12-2000) : get local grid size from definition of g.
!>                         Assume that gsq is allocated.
!>                         Local routine, no information on distribution of
!>                         PW required.
!>      JGH (8-Mar-2001) : also update information on volume and grid spaceing
!> \author apsi
!>      Christopher Mundy
! **************************************************************************************************
   SUBROUTINE pw_grid_change(cell_hmat, pw_grid)
      ! Argument
      REAL(KIND=dp), DIMENSION(3, 3), INTENT(IN)         :: cell_hmat
      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid

      INTEGER                                            :: gpt
      REAL(KIND=dp)                                      :: cell_deth, l, m, n
      REAL(KIND=dp), DIMENSION(3, 3)                     :: cell_h_inv
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: g

      cell_deth = ABS(det_3x3(cell_hmat))
      IF (cell_deth < 1.0E-10_dp) THEN
         CALL cp_abort(__LOCATION__, &
                       "An invalid set of cell vectors was specified. "// &
                       "The determinant det(h) is too small")
      END IF
      cell_h_inv = inv_3x3(cell_hmat)

      g => pw_grid%g

      ! the g-vectors change in place: drop device copies cached by pw_gpu
      CALL pw_grid_gpu_invalidate(pw_grid%id_nr)

      CALL cell2grid(cell_hmat, cell_h_inv, cell_deth, pw_grid)

      DO gpt = 1, SIZE(g, 2)

         l = twopi*REAL(pw_grid%g_hat(1, gpt), KIND=dp)
         m = twopi*REAL(pw_grid%g_hat(2, gpt), KIND=dp)
         n = twopi*REAL(pw_grid%g_hat(3, gpt), KIND=dp)

         g(1, gpt) = l*cell_h_inv(1, 1) + m*cell_h_inv(2, 1) + n*cell_h_inv(3, 1)
         g(2, gpt) = l*cell_h_inv(1, 2) + m*cell_h_inv(2, 2) + n*cell_h_inv(3, 2)
         g(3, gpt) = l*cell_h_inv(1, 3) + m*cell_h_inv(2, 3) + n*cell_h_inv(3, 3)

         pw_grid%gsq(gpt) = g(1, gpt)*g(1, gpt) &
                            + g(2, gpt)*g(2, gpt) &
                            + g(3, gpt)*g(3, gpt)

      END DO

   END SUBROUTINE pw_grid_change

! **************************************************************************************************
!> \brief retains the given pw grid
!> \param pw_grid the pw grid to retain
!> \par History
!>      04.2003 created [fawzi]
!> \author fawzi
!> \note
!>      see doc/ReferenceCounting.html
! **************************************************************************************************
   SUBROUTINE pw_grid_retain(pw_grid)
      TYPE(pw_grid_type), INTENT(INOUT)                  :: pw_grid

      CPASSERT(pw_grid%ref_count > 0)
      pw_grid%ref_count = pw_grid%ref_count + 1
   END SUBROUTINE pw_grid_retain

! **************************************************************************************************
!> \brief releases the given pw grid
!> \param pw_grid the pw grid to release
!> \par History
!>      04.2003 created [fawzi]
!> \author fawzi
!> \note
!>      see doc/ReferenceCounting.html
! **************************************************************************************************
   SUBROUTINE pw_grid_release(pw_grid)

      TYPE(pw_grid_type), POINTER              :: pw_grid

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
      INTEGER, POINTER :: dummy_ptr
      INTEGER          :: stat
#endif

      IF (ASSOCIATED(pw_grid)) THEN
         CPASSERT(pw_grid%ref_count > 0)
         pw_grid%ref_count = pw_grid%ref_count - 1
         IF (pw_grid%ref_count == 0) THEN
            ! free device copies cached by pw_gpu for this grid
            CALL pw_grid_gpu_invalidate(pw_grid%id_nr)
            IF (ASSOCIATED(pw_grid%gidx)) THEN
               DEALLOCATE (pw_grid%gidx)
            END IF
            IF (ASSOCIATED(pw_grid%g)) THEN
               DEALLOCATE (pw_grid%g)
            END IF
            IF (ASSOCIATED(pw_grid%gsq)) THEN
               DEALLOCATE (pw_grid%gsq)
            END IF
            IF (ALLOCATED(pw_grid%g_hat)) THEN
               DEALLOCATE (pw_grid%g_hat)
            END IF
            IF (ASSOCIATED(pw_grid%g_hatmap)) THEN
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
               dummy_ptr => pw_grid%g_hatmap(1, 1)
               stat = offload_free_pinned_mem(c_loc(dummy_ptr))
               CPASSERT(stat == 0)
#else
               DEALLOCATE (pw_grid%g_hatmap)
#endif
            END IF
            IF (ASSOCIATED(pw_grid%grays)) THEN
               DEALLOCATE (pw_grid%grays)
            END IF
            IF (ALLOCATED(pw_grid%mapl%pos)) THEN
               DEALLOCATE (pw_grid%mapl%pos)
            END IF
            IF (ALLOCATED(pw_grid%mapm%pos)) THEN
               DEALLOCATE (pw_grid%mapm%pos)
            END IF
            IF (ALLOCATED(pw_grid%mapn%pos)) THEN
               DEALLOCATE (pw_grid%mapn%pos)
            END IF
            IF (ALLOCATED(pw_grid%mapl%neg)) THEN
               DEALLOCATE (pw_grid%mapl%neg)
            END IF
            IF (ALLOCATED(pw_grid%mapm%neg)) THEN
               DEALLOCATE (pw_grid%mapm%neg)
            END IF
            IF (ALLOCATED(pw_grid%mapn%neg)) THEN
               DEALLOCATE (pw_grid%mapn%neg)
            END IF
            IF (ALLOCATED(pw_grid%para%bo)) THEN
               DEALLOCATE (pw_grid%para%bo)
            END IF
            IF (pw_grid%para%mode == PW_MODE_DISTRIBUTED) THEN
               IF (ALLOCATED(pw_grid%para%yzp)) THEN
                  DEALLOCATE (pw_grid%para%yzp)
               END IF
               IF (ALLOCATED(pw_grid%para%yzq)) THEN
                  DEALLOCATE (pw_grid%para%yzq)
               END IF
               IF (ALLOCATED(pw_grid%para%nyzray)) THEN
                  DEALLOCATE (pw_grid%para%nyzray)
               END IF
            END IF
            ! also release groups
            CALL pw_grid%para%group%free()
            IF (ALLOCATED(pw_grid%para%pos_of_x)) THEN
               DEALLOCATE (pw_grid%para%pos_of_x)
            END IF

            IF (ASSOCIATED(pw_grid)) THEN
               DEALLOCATE (pw_grid)
            END IF
         END IF
      END IF
      NULLIFY (pw_grid)
   END SUBROUTINE pw_grid_release

END MODULE pw_grids
