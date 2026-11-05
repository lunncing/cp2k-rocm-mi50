! SPDX-License-Identifier: GPL-2.0-or-later
! Production routines and bindings are extracted by grid_ops_tests.py.
#:set pw_kinds = ['r1d', 'c1d', 'r3d', 'c3d']
#:set pw_spaces = ['rs', 'gs']
#:set pairs = [(a, b) for a in pw_kinds for b in pw_kinds if a[1:]==b[1:]]
#:set declaration = {'r': 'REAL(KIND=dp)', 'c': 'COMPLEX(KIND=dp)'}
#:set dims = {'1': ':', '3': ':, :, :'}
#:set type2type = lambda v,a,b: v if a[0]==b[0] else ('REAL('+v+',KIND=dp)' if b[0]=='r' &
                 & else 'CMPLX('+v+',0.0_dp,KIND=dp)')
#define CPABORT(message) ERROR STOP message
#define MARK_USED(value)
MODULE grid_fixture
   USE ISO_C_BINDING
   USE ISO_FORTRAN_ENV, ONLY: int64
   USE, INTRINSIC :: ieee_arithmetic
   IMPLICIT NONE
   INTEGER, PARAMETER :: dp = C_DOUBLE, xc_pbe_orig = 1, xc_pbe_rev = 2, xc_pbe_sol = 3
   REAL(KIND=dp), PARAMETER :: pi = 3.141592653589793238462643383279502884_dp
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
   LOGICAL, PARAMETER :: gpu_enabled = .TRUE.
#else
   LOGICAL, PARAMETER :: gpu_enabled = .FALSE.
#endif
   LOGICAL :: limited
   TYPE grid_type
      INTEGER :: id_nr = 1, reference = 0, first_gne0 = 2, ngpts_cut_local = 0
      LOGICAL :: spherical = .FALSE., have_g0 = .TRUE., compatible = .TRUE.
      REAL(KIND=dp) :: vol = 27.3_dp
      INTEGER, POINTER :: gidx(:) => NULL()
      REAL(KIND=dp), POINTER :: g(:, :) => NULL(), gsq(:) => NULL()
   END TYPE
   TYPE particle_type
      REAL(KIND=dp) :: r(3)
   END TYPE
   TYPE section_vals_type
      REAL(KIND=dp) :: gcut
   END TYPE
   #:for space in pw_spaces
      #:for kind in pw_kinds
         TYPE pw_${kind}$_${space}$_type
            ${declaration[kind[0]]}$, POINTER :: array(${dims[kind[1]]}$) => NULL()
            TYPE(grid_type), POINTER :: pw_grid => NULL()
         END TYPE
      #:endfor
   #:endfor
   INTERFACE pw_zero
      #:for space in pw_spaces
         #:for kind in pw_kinds
            MODULE PROCEDURE zero_${kind}$_${space}$
         #:endfor
      #:endfor
   END INTERFACE
   INTERFACE
      SUBROUTINE test_gpu_begin(limited) BIND(C)
         IMPORT :: C_INT
         INTEGER(C_INT), VALUE :: limited
      END SUBROUTINE
      SUBROUTINE test_gpu_end() BIND(C)
      END SUBROUTINE
      FUNCTION test_kernel_calls(kind) BIND(C) RESULT(n)
         IMPORT :: C_INT
         INTEGER(C_INT), VALUE :: kind
         INTEGER(C_INT) :: n
      END FUNCTION
   END INTERFACE
CONTAINS
   #:include "grid_bindings.inc"
   SUBROUTINE timeset(name, handle)
      CHARACTER(LEN=*), INTENT(IN) :: name
      INTEGER, INTENT(OUT) :: handle
      handle = LEN_TRIM(name)
   END SUBROUTINE
   SUBROUTINE timestop(handle)
      INTEGER, INTENT(IN) :: handle
      IF (handle <= 0) ERROR STOP 'invalid timer'
   END SUBROUTINE
   LOGICAL FUNCTION pw_compatible(a, b)
      TYPE(grid_type), INTENT(IN) :: a, b
      pw_compatible = a%compatible .AND. b%compatible
   END FUNCTION
   SUBROUTINE section_vals_val_get(section, name, r_val)
      TYPE(section_vals_type), POINTER :: section
      CHARACTER(LEN=*), INTENT(IN) :: name
      REAL(KIND=dp), INTENT(OUT) :: r_val
      IF (name /= 'GCUT') ERROR STOP 'unexpected keyword'
      r_val = section%gcut
   END SUBROUTINE
   SUBROUTINE pw_copy_match(a, b)
      TYPE(pw_c1d_gs_type), INTENT(IN) :: a
      TYPE(pw_c1d_gs_type), INTENT(INOUT) :: b
      ERROR STOP 'unexpected copy match'
   END SUBROUTINE
   #:for space in pw_spaces
      #:for kind in pw_kinds
         SUBROUTINE zero_${kind}$_${space}$ (a)
            TYPE(pw_${kind}$_${space}$_type), INTENT(INOUT) :: a
            a%array = 0.0_dp
         END SUBROUTINE
      #:endfor
   #:endfor
   #:for kind, kind2 in pairs
      #:for space in pw_spaces
         #:include "copy.inc"
         #:include "axpy.inc"
      #:endfor
   #:endfor
   ! modify_hartree_pot references the device-resident variant (residency stage 1);
   ! it is only called with the optional device_done argument, which these tests omit.
   LOGICAL FUNCTION pw_gpu_res_hartree_modify(g0_delta, grid, positions, radii, cv, gcut2, vol) RESULT(done)
      REAL(KIND=dp), INTENT(IN) :: g0_delta, gcut2, vol
      TYPE(grid_type), INTENT(IN) :: grid
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN) :: positions
      REAL(KIND=dp), DIMENSION(:), INTENT(IN) :: radii, cv
      done = .FALSE.
      IF (g0_delta == gcut2 .AND. vol < 0.0_dp .AND. grid%id_nr < 0 .AND. SIZE(positions) + SIZE(radii) + &
          SIZE(cv) < 0) done = .TRUE.
      ERROR STOP 'pw_gpu_res_hartree_modify is not used by these tests'
   END FUNCTION
   #:include "pbe_lda_calc.inc"
   #:include "modify_hartree_pot.inc"
END MODULE

PROGRAM test_grid_operations
   USE grid_fixture
   IMPLICIT NONE
   INTEGER :: phase
   DO phase = 0, 1
      limited = phase == 1
      CALL test_gpu_begin(INT(phase, C_INT))
      #:for kind, kind2 in pairs
         #:for space in pw_spaces
            CALL grid_${kind}$_${kind2}$_${space}$ ()
         #:endfor
      #:endfor
      CALL hartree()
      CALL pbe()
      CALL test_gpu_end()
   END DO
   PRINT '(a)', 'PASS grid copy/AXPY bitwise, Hartree/PBE <=1e-12, CPU fallback and bindings'
CONTAINS
   SUBROUTINE close(a, b, what)
      REAL(KIND=dp), INTENT(IN) :: a(:), b(:)
      CHARACTER(LEN=*), INTENT(IN) :: what
      IF (.NOT. ALL(ABS(a - b) <= 1.e-12_dp*MAX(1.0_dp, ABS(b)))) THEN
         PRINT *, what, MAXVAL(ABS(a - b)), MAXLOC(ABS(a - b))
         ERROR STOP 'CPU/GPU numerical mismatch'
      END IF
   END SUBROUTINE
   #:for kind, kind2 in pairs
      #:for space in pw_spaces
         SUBROUTINE grid_${kind}$_${kind2}$_${space}$ ()
            TYPE(pw_${kind}$_${space}$_type) :: a, ar
            TYPE(pw_${kind2}$_${space}$_type) :: b, br
            TYPE(grid_type), TARGET :: g1, g2
            ${declaration[kind[0]]}$, ALLOCATABLE, TARGET :: xs(${dims[kind[1]]}$), xr(${dims[kind[1]]}$)
            ${declaration[kind2[0]]}$, ALLOCATABLE, TARGET :: ys(${dims[kind[1]]}$), yr(${dims[kind[1]]}$)
            ${declaration[kind[0]]}$, ALLOCATABLE :: x0(${dims[kind[1]]}$)
            ${declaration[kind2[0]]}$, ALLOCATABLE :: y0(${dims[kind[1]]}$)
            REAL(KIND=dp), PARAMETER :: factors(*) = [1.0_dp, 0.0_dp, -1.23_dp, 0.37_dp]
            INTEGER :: layout, nx, ny, stride, i, ia, ib, before
            DO layout = 0, 9
               #:if kind[1] == '3'
                  IF (layout >= 1 .AND. layout <= 5) CYCLE
               #:endif
               #:if kind[1] == '1'
                  IF (layout == 9) CYCLE
               #:endif
               #:if space == 'rs'
                  IF (layout == 5) CYCLE
               #:endif
               #:if kind != kind2
                  IF (layout == 8) CYCLE
               #:endif
               nx = 17
               ny = 17
               IF (layout == 1 .OR. layout == 3) ny = 7
               IF (layout == 2 .OR. layout == 4 .OR. layout == 5) nx = 7
               IF (layout == 7) THEN
                  nx = 0
                  ny = 0
               END IF
               stride = MERGE(2, 1, layout == 6)
               #:if kind[1] == '1'
                  ALLOCATE (xs(stride*nx), xr(stride*nx), ys(stride*ny), yr(stride*ny))
                  a%array => xs(1:stride*nx:stride)
                  ar%array => xr(1:stride*nx:stride)
                  b%array => ys(1:stride*ny:stride)
                  br%array => yr(1:stride*ny:stride)
               #:else
                  ALLOCATE (xs(stride*nx, 2, 1), xr(stride*nx, 2, 1), ys(stride*ny, 2, 1), yr(stride*ny, 2, 1))
                  a%array => xs(1:stride*nx:stride, :, :)
                  ar%array => xr(1:stride*nx:stride, :, :)
                  b%array => ys(1:stride*ny:stride, :, :)
                  br%array => yr(1:stride*ny:stride, :, :)
               #:endif
               #:if kind == kind2
                  IF (layout == 8) THEN
                     b%array => a%array
                     br%array => ar%array
                  END IF
               #:endif
               g1%spherical = layout == 5
               g2%spherical = layout == 5
               g1%reference = MERGE(2, 0, layout == 3 .OR. layout == 4)
               g2%reference = MERGE(1, 0, layout == 1 .OR. layout == 2)
               g2%id_nr = 2
               ALLOCATE (g1%gidx(MIN(nx, ny)), g2%gidx(MIN(nx, ny)))
               g1%gidx = [(2*i - 1, i=1, MIN(nx, ny))]
               g2%gidx = g1%gidx
               a%pw_grid => g1
               ar%pw_grid => g1
               b%pw_grid => g1
               IF ((layout >= 1 .AND. layout <= 5) .OR. layout == 9) b%pw_grid => g2
               br%pw_grid => b%pw_grid
               xs = RESHAPE([(SIN(i*0.17_dp), i=1, SIZE(xs))], SHAPE(xs))
               ys = RESHAPE([(COS(i*0.13_dp), i=1, SIZE(ys))], SHAPE(ys))
               #:if kind[0] == 'c'
                  xs = CMPLX(REAL(xs, dp), REAL(xs, dp)*(-0.37_dp), dp)
               #:endif
               #:if kind2[0] == 'c'
                  ys = CMPLX(REAL(ys, dp), REAL(ys, dp)*0.19_dp, dp)
               #:endif
               IF (layout == 9) xs = ieee_value(0.0_dp, ieee_quiet_nan)
               x0 = xs
               y0 = ys
               xr = xs
               yr = ys
               before = test_kernel_calls(0_C_INT)
               CALL pw_copy_${kind}$_${kind2}$_${space}$ (a, b)
               CALL cpu_copy_${kind}$_${kind2}$_${space}$ (ar, br)
               IF (ANY(TRANSFER(b%array, [0_int64]) /= TRANSFER(br%array, [0_int64]))) ERROR STOP 'copy bits'
               IF (test_kernel_calls(0_C_INT) /= before) ERROR STOP 'unexpected copy offload'
               DO ia = 1, SIZE(factors)
                  DO ib = 1, SIZE(factors)
                     xs = x0
                     ys = y0
                     xr = x0
                     yr = y0
                     CALL pw_axpy_${kind}$_${kind2}$_${space}$ (a, b, factors(ia), factors(ib))
                     CALL cpu_axpy_${kind}$_${kind2}$_${space}$ (ar, br, factors(ia), factors(ib))
                     IF (test_kernel_calls(0_C_INT) /= before) ERROR STOP 'unexpected AXPY offload'
                     IF (ANY(TRANSFER(b%array, [0_int64]) /= TRANSFER(br%array, [0_int64]))) THEN
                        PRINT *, '${kind}$ ${kind2}$ ${space}$', layout, factors(ia), factors(ib)
                        ERROR STOP 'AXPY bits'
                     END IF
                  END DO
               END DO
               DEALLOCATE (xs, xr, ys, yr, x0, y0, g1%gidx, g2%gidx)
            END DO
         END SUBROUTINE
      #:endfor
   #:endfor

   SUBROUTINE hartree()
      INTEGER, PARAMETER :: n = 263, na = 5, nr = 3
      TYPE(pw_c1d_gs_type) :: v, ref
      TYPE(grid_type), TARGET :: grid
      TYPE(particle_type), POINTER :: atoms(:)
      TYPE(section_vals_type), POINTER :: section
      REAL(KIND=dp), POINTER :: m(:, :), ami(:, :), radii(:), charges(:)
      INTEGER :: i, j, g0, before
      ALLOCATE (v%array(n), ref%array(n), grid%g(3, n), grid%gsq(n), atoms(na), section)
      ALLOCATE (m(na*nr, na), ami(na*nr, na*nr), radii(nr), charges(na))
      v%pw_grid => grid
      ref%pw_grid => grid
      radii = [.1_dp, .7_dp, 2.3_dp]
      charges = [(SIN(REAL(i, dp)), i=1, na)]
      DO i = 1, na
         atoms(i)%r = [.17_dp*i, -.3_dp*i, .8_dp*i]
         m(:, i) = [(COS(j*0.2_dp + i), j=1, na*nr)]
      END DO
      ami = .01_dp
      DO i = 1, na*nr
         ami(i, i) = 2.0_dp
      END DO
      grid%ngpts_cut_local = n
      grid%gsq = [(REAL(i - 1, dp)*.01_dp, i=1, n)]
      DO i = 1, n
         grid%g(:, i) = [SIN(i*.31_dp), COS(i*.13_dp), SIN(i*.41_dp)]
      END DO
      section%gcut = SQRT(grid%gsq(n - 7))
      DO g0 = 0, 1
         grid%have_g0 = g0 == 1
         grid%first_gne0 = 2
         v%array = [(CMPLX(SIN(i*.2_dp), COS(i*.7_dp), dp), i=1, n)]
         ref%array = v%array
         before = test_kernel_calls(1_C_INT)
         CALL modify_hartree_pot(v, section, atoms, m, ami, radii, charges)
         CALL cpu_modify_hartree_pot(ref, section, atoms, m, ami, radii, charges)
         CALL close(REAL(v%array, dp), REAL(ref%array, dp), 'Hartree real')
         CALL close(AIMAG(v%array), AIMAG(ref%array), 'Hartree imaginary')
         IF ((test_kernel_calls(1_C_INT) > before) .NEQV. (gpu_enabled .AND. .NOT. limited)) &
            ERROR STOP 'Hartree dispatch'
         IF (ANY(v%array(n - 5:n) /= ref%array(n - 5:n))) ERROR STOP 'Hartree cutoff'
      END DO
      grid%first_gne0 = 1
      grid%ngpts_cut_local = 0
      grid%have_g0 = .FALSE.
      CALL modify_hartree_pot(v, section, atoms, m, ami, radii, charges)
      DEALLOCATE (v%array, ref%array, grid%g, grid%gsq, atoms, section, m, ami, radii, charges)
   END SUBROUTINE

   SUBROUTINE pbe()
      INTEGER, PARAMETER :: n = 263
      REAL(KIND=dp) :: rho(n), norm(n), e(n, 10), ref(n, 10), eps, sc, sx
      INTEGER :: i, grad, param, scaling, before, nuse
      eps = 1.e-10_dp
      rho = [(10.0_dp**(-11.0_dp + 14.0_dp*i/n), i=1, n)]
      rho(1:5) = [-1.0_dp, 0.0_dp, eps, NEAREST(eps, -1.0_dp), NEAREST(eps, 1.0_dp)]
      rho(6) = ieee_value(0.0_dp, ieee_quiet_nan)
      norm = [(rho(i)**(4.0_dp/3)*ABS(SIN(i*.3_dp)), i=1, n)]
      norm(1:6) = 0.0_dp
      norm(20:30) = 0.0_dp
      DO param = 1, 3
         DO grad = 0, 3
            DO scaling = 0, 3
               sc = MERGE(1.0_dp, .37_dp, scaling == 0)
               sx = MERGE(1.0_dp, -.21_dp, scaling == 0)
               IF (scaling == 1) sc = 0.0_dp
               IF (scaling == 2) sx = 0.0_dp
               nuse = MERGE(0, n, scaling == 3)
               e = .17_dp
               ref = e
               before = test_kernel_calls(2_C_INT)
               #:for call, array in [('pbe_lda_calc','e'), ('cpu_pbe_lda_calc','ref')]
!$OMP PARALLEL
                  CALL ${call}$(rho, norm, ${array}$(:, 1), ${array}$(:, 2), ${array}$(:, 3), &
                                ${array}$(:, 4), ${array}$(:, 5), ${array}$(:, 6), ${array}$(:, 7), &
                                ${array}$(:, 8), ${array}$(:, 9), ${array}$(:, 10), grad, nuse, eps, param, sc, sx)
!$OMP END PARALLEL
               #:endfor
               DO i = 1, 10
                  CALL close(e(:, i), ref(:, i), 'PBE energy/derivatives')
               END DO
               IF (ANY(e(1:4, :) /= .17_dp) .OR. ANY(e(6, :) /= .17_dp)) ERROR STOP 'PBE cutoff changed'
               IF (grad <= 1 .AND. nuse > 0 .AND. gpu_enabled .AND. .NOT. limited) THEN
                  IF (test_kernel_calls(2_C_INT) /= before + 1) ERROR STOP 'PBE must launch exactly once per OMP team'
               ELSE
                  IF (test_kernel_calls(2_C_INT) /= before) ERROR STOP 'unexpected PBE offload'
               END IF
            END DO
         END DO
      END DO
      ! pbe_lda_eval aliases all unused derivative pointers to rho for grad=0.
      ref(:, 1) = rho
      e(:, 1) = .17_dp
      before = test_kernel_calls(2_C_INT)
!$OMP PARALLEL
      CALL pbe_lda_calc(rho, norm, e(:, 1), rho, rho, rho, rho, rho, rho, rho, rho, rho, &
                        0, n, eps, xc_pbe_orig, 1.0_dp, 1.0_dp)
!$OMP END PARALLEL
      IF (ANY(TRANSFER(rho, [0_int64]) /= TRANSFER(ref(:, 1), [0_int64]))) &
         ERROR STOP 'PBE overwrote aliased unused derivatives'
      IF ((test_kernel_calls(2_C_INT) > before) .NEQV. (gpu_enabled .AND. .NOT. limited)) &
         ERROR STOP 'energy-only PBE dispatch'
   END SUBROUTINE
END PROGRAM
