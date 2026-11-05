/* SPDX-License-Identifier: GPL-2.0-or-later */
#include <assert.h>
#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

double pw_cpu_dot_complex_c(const void *a, const void *b, int n_complex);

static uint32_t random_state = 17321;

static double random_value(void) {
  random_state = 1664525U * random_state + 1013904223U;
  return (double)(random_state >> 8) / 8388608.0 - 1.0;
}

static void check(const double complex *a, const double complex *b, int n) {
  long double expected = 0.0L, magnitude = 0.0L;
  for (int i = 0; i < n; i++) {
    const long double complex product = conjl((long double complex)a[i]) * b[i];
    expected += creall(product);
    magnitude += fabsl((long double)creal(a[i]) * creal(b[i])) +
                 fabsl((long double)cimag(a[i]) * cimag(b[i]));
  }
  const double actual = pw_cpu_dot_complex_c(a, b, n);
  // BLAS changes the summation order. Use the forward-error bound for a dot
  // product of 2*n doubles, scaled by the magnitude of all products.
  const long double roundoff = (2.0L * n + 1) * DBL_EPSILON;
  const long double tolerance = roundoff / (1 - roundoff) * fmaxl(1, magnitude);
  if (!(fabsl(actual - expected) <= tolerance)) {
    fprintf(stderr, "Complex dot mismatch for n=%d: %.17g != %.21Lg\n",
            n, actual, expected);
    abort();
  }
}

int main(void) {
  const int sizes[] = {0, 1, 2, 3, 7, 31, 64, 257, 1023, 4096, 65537};
  double complex *a = malloc(65538 * sizeof(*a));
  double complex *b = malloc(65538 * sizeof(*b));
  assert(a != NULL && b != NULL);
  check(NULL, NULL, 0);
  // The imaginary components must contribute with a positive sign.
  a[0] = 2 + 3 * I;
  b[0] = 5 + 7 * I;
  assert(pw_cpu_dot_complex_c(a, b, 1) == 31.0);
  a[0] = 3 * I;
  b[0] = -7 * I;
  assert(pw_cpu_dot_complex_c(a, b, 1) == -21.0);
  for (int trial = 0; trial < 8; trial++) {
    for (int i = 0; i < 65538; i++) {
      const double ar = random_value(), ai = random_value();
      const double br = random_value(), bi = random_value();
      a[i] = ldexp(ar, trial - 4) + I * ldexp(ai, 4 - trial);
      b[i] = ldexp(br, 4 - trial) + I * ldexp(bi, trial - 4);
    }
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
      check(a, b, sizes[i]);
      check(a + 1, b + 1, sizes[i]); // Non-vector-aligned input.
      check(a, a, sizes[i]);         // Aliased inputs / squared norm.
    }
  }
  free(a);
  free(b);
  puts("PASS complex DDOT random vectors, empty/singleton inputs, offsets, and norms");
  return 0;
}
