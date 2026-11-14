"""Compile the production mixing loops and check thread-count determinism."""
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]

PRELUDE = """
MODULE mixing_loop_fixture
   USE ISO_FORTRAN_ENV, ONLY: int64, real64
   USE OMP_LIB, ONLY: omp_set_num_threads
   IMPLICIT NONE
   INTEGER, PARAMETER :: dp = real64, grid_n = 257, nb = 3, result_size = 8192
   COMPLEX(dp), PARAMETER :: z_zero = (0.0_dp, 0.0_dp)
   TYPE complex_grid
      COMPLEX(dp) :: cc(grid_n)
   END TYPE
   TYPE pw_type
      COMPLEX(dp) :: array(grid_n)
   END TYPE
   TYPE mixing_type
      TYPE(complex_grid) :: rhoin(2), rhoin_old(2), last_res(2)
      TYPE(complex_grid) :: res_buffer(nb, 2), rhoin_buffer(nb, 2), drho_buffer(nb, 2)
      INTEGER :: ig_max = grid_n - 5
      REAL(dp) :: p_metric(grid_n), special_metric(grid_n), pulay_beta = 0.31_dp, pulay_matrix(nb, nb, 2)
   END TYPE
CONTAINS
   COMPLEX(dp) FUNCTION seed(point, spin)
      INTEGER, INTENT(IN) :: point, spin
      seed = CMPLX(SIN(0.17_dp*point*spin), COS(0.23_dp*point*spin), dp)
   END FUNCTION
   SUBROUTINE initialize(mixing_store, rho_g)
      TYPE(mixing_type), INTENT(OUT) :: mixing_store
      TYPE(pw_type), INTENT(OUT) :: rho_g(2)
      INTEGER :: point, spin, buffer
      mixing_store%ig_max = grid_n - 5
      mixing_store%pulay_matrix = 0
      DO point = 1, grid_n
         mixing_store%p_metric(point) = 0.2_dp + 0.001_dp*point
         mixing_store%special_metric(point) = 0.7_dp + 0.002_dp*point
         DO spin = 1, 2
            rho_g(spin)%array(point) = seed(point, spin)
            mixing_store%rhoin(spin)%cc(point) = 0.71_dp*seed(point, spin)
            mixing_store%rhoin_old(spin)%cc(point) = 0.41_dp*seed(point, spin)
            mixing_store%last_res(spin)%cc(point) = 0.13_dp*seed(point, spin)
            DO buffer = 1, nb
               mixing_store%res_buffer(buffer, spin)%cc(point) = seed(point, spin + buffer)*0.03_dp
               mixing_store%rhoin_buffer(buffer, spin)%cc(point) = seed(point, spin + buffer)*0.63_dp
               mixing_store%drho_buffer(buffer, spin)%cc(point) = seed(point, spin + buffer)*0.17_dp
            END DO
         END DO
      END DO
   END SUBROUTINE
"""

SETUP = """
      INTEGER, INTENT(IN) :: threads
      COMPLEX(dp), INTENT(OUT) :: result(result_size)
      TYPE(mixing_type) :: mixing_store
      TYPE(pw_type), TARGET :: rho_g(2)
      INTEGER :: ig, jb, kb, ispin, ib, ibb, point, spin, buffer, offset, ng
      COMPLEX(dp) :: c1, c2, res_rho(grid_n)
      COMPLEX(dp), POINTER :: cc_new(:)
      REAL(dp) :: f_mix, alpha_eff, alpha_kerker, alpha_pulay, alpha_c(nb), kerker_eff(grid_n)
      REAL(dp) :: a(nb, nb), c(nb), g(nb), w0, res_norm
      CALL omp_set_num_threads(threads)
      CALL initialize(mixing_store, rho_g)
      ng = grid_n
      ispin = 2
      ib = 1
      ibb = 2
      jb = 2
      cc_new => rho_g(ispin)%array
      alpha_eff = 0.17_dp
      alpha_kerker = 0.23_dp
      alpha_pulay = 0.37_dp
      alpha_c = [0.2_dp, 0.5_dp, 0.3_dp]
      g = [0.1_dp, -0.03_dp, 0.07_dp]
      kerker_eff = mixing_store%p_metric
      cc_mix = z_zero
      a = 0
      c = 0
      w0 = 0.01_dp
      res_norm = 1.0_dp
      DO point = 1, ng
         res_rho(point) = seed(point, 3)*0.05_dp
      END DO
"""

COLLECT = """
      result = z_zero
      result(1:ng) = rho_g(1)%array
      result(ng + 1:2*ng) = rho_g(2)%array
      offset = 2*ng
      DO spin = 1, 2
         result(offset + 1:offset + ng) = mixing_store%rhoin(spin)%cc
         result(offset + ng + 1:offset + 2*ng) = mixing_store%rhoin_old(spin)%cc
         result(offset + 2*ng + 1:offset + 3*ng) = mixing_store%last_res(spin)%cc
         offset = offset + 3*ng
         DO buffer = 1, nb
            result(offset + 1:offset + ng) = mixing_store%res_buffer(buffer, spin)%cc
            result(offset + ng + 1:offset + 2*ng) = mixing_store%rhoin_buffer(buffer, spin)%cc
            result(offset + 2*ng + 1:offset + 3*ng) = mixing_store%drho_buffer(buffer, spin)%cc
            offset = offset + 3*ng
         END DO
      END DO
      result(offset + 1:offset + ng) = res_rho
      offset = offset + ng
      result(offset + 1:offset + nb*nb) = CMPLX(RESHAPE(a, [nb*nb]), 0.0_dp, dp)
      offset = offset + nb*nb
      result(offset + 1:offset + nb) = CMPLX(c, 0.0_dp, dp)
      offset = offset + nb
      result(offset + 1:offset + 2*nb*nb) = CMPLX(RESHAPE(mixing_store%pulay_matrix, [2*nb*nb]), 0.0_dp, dp)
      offset = offset + 2*nb*nb
"""


def run_mixing_tests(build, env, sanitize):
    source = (ROOT / "src/qs_gspace_mixing.F").read_text()
    blocks = list(re.finditer(r"!\$OMP PARALLEL DO.*?!\$OMP END PARALLEL DO", source, re.S))
    assert len(blocks) == 20
    code = [PRELUDE]
    checks = []
    for index, match in enumerate(blocks):
        owner = re.findall(r"^   SUBROUTINE (\w+)\(", source[:match.start()], re.M)[-1]
        array_mix = owner == "pulay_mixing"
        for mode in ("serial", "parallel"):
            name = f"{mode}_{index}"
            setup = SETUP.replace("      CALL omp_set_num_threads(threads)",
                                  "      COMPLEX(dp) :: cc_mix" + ("(grid_n)" if array_mix else "") +
                                  "\n      CALL omp_set_num_threads(threads)")
            body = match[0]
            if mode == "serial":
                body = re.sub(r"^!\$OMP.*$", "", body, flags=re.M)
            code += [f"   SUBROUTINE {name}(threads, result)", setup, body, COLLECT]
            if array_mix:
                code += ["      result(offset + 1:offset + ng) = cc_mix"]
            code += [f"   END SUBROUTINE {name}"]
        checks += [f"   CALL serial_{index}(1, expected)",
                   f"   CALL parallel_{index}(3, actual)",
                   f"   IF (ANY(TRANSFER(actual, bits) /= TRANSFER(expected, bits))) ERROR STOP 'mixing loop {index}'"]
        if index in (0, 2, 3):
            checks += ["   DO point = 1, ng"]
            if index == 0:
                checks += ["      first = seed(point, 1) + seed(point, 2)",
                           "      second = seed(point, 1) - seed(point, 2)"]
            else:
                checks += ["      first = 0.5_dp*(seed(point, 1) + seed(point, 2))",
                           "      second = first - seed(point, 2)"]
            checks += ["      IF (actual(point) /= first .OR. actual(ng + point) /= second) &",
                       "         ERROR STOP 'spin conversion arithmetic'", "   END DO"]
    code += ["END MODULE", "PROGRAM test_mixing_loops", "   USE mixing_loop_fixture", "   IMPLICIT NONE",
             "   COMPLEX(dp) :: actual(result_size), expected(result_size), first, second",
             "   INTEGER(int64) :: bits(2*result_size)", "   INTEGER :: point, ng", "   ng = grid_n",
             *checks, "   PRINT *, 'PASS mixing loops: deterministic OMP=1/3 and spin conversions'", "END PROGRAM"]
    path = build / "test_mixing_loops.F90"
    text = "\n".join(code) + "\n"
    assert max(map(len, text.splitlines())) <= 132
    path.write_text(text)
    fc = shlex.split(os.environ.get("FC", "gfortran"))
    flags = ["-std=f2008", "-fopenmp", "-ffp-contract=off", "-fcheck=all", "-Werror=line-truncation",
             "-J", str(build), "-O1", "-g"]
    if sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    exe = build / "test_mixing_loops"
    subprocess.run(fc + flags + [str(path), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], env=env, check=True, timeout=60)
