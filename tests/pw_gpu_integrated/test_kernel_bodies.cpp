/* SPDX-License-Identifier: GPL-2.0-or-later */
// Appended to the emulated production .cu translation unit by the test runner.
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
  for (size_t n : {size_t(0), size_t(1), size_t(255), size_t(256), size_t(257), size_t(1025)}) {
    std::vector<double> x(2*n), y(2*n, -7), expected(2*n);
    std::vector<int> map(n);
    for (size_t i = 0; i < 2*n; i++) x[i] = sin(.17*i);
    for (size_t i = 0; i < n; i++) map[i] = (int)(n - i);
    pw_gpu_launch_copy(x.data(), y.data(), n, n, 2, 2, 1.0, 0);
    assert(y == x);
    pw_gpu_launch_axpy(x.data(), y.data(), n, n, 2, 2, .3, -.7, 0);
    for (size_t i = 0; i < 2*n; i++) expected[i] = -.7*x[i] + .3*x[i];
    assert(y == expected);
    for (int scatter = 0; scatter < 2; scatter++) {
      pw_gpu_launch_remap_copy(x.data(), y.data(), map.data(), n, 2, 2, scatter, 1.0, 0);
      for (size_t i = 0; i < n; i++) for (int c = 0; c < 2; c++)
        expected[2*i+c] = x[2*(n-1-i)+c];
      assert(y == expected);
      pw_gpu_launch_remap_axpy(x.data(), y.data(), map.data(), n, 2, 2, scatter, 1.0, .5, 0);
      for (double &v : expected) v = .5*v + v;
      assert(y == expected);
    }
    // A deliberately small launch exercises the production grid-stride loop.
    emulate(pw_gpu_copy_kernel, 2, 7, 0, 0, x.data(), y.data(), n, n, 2, 2, 1.0);
    assert(y == x);
    std::vector<double> rho(n, .7), norm(n, .2), e(n, 0), er(n, 0), en(n, 0);
    std::vector<double> ref(n, 0), rr(n, 0), rn(n, 0);
    pw_gpu_launch_pbe_lda(rho.data(), norm.data(), e.data(), er.data(), en.data(), n,
                          1, 1e-12, .804, .066725, .21951, 1.0, 1.0, 0);
    for (size_t i = 0; i < n; i++)
      pw_pbe_lda_elem(rho.data(), norm.data(), ref.data(), rr.data(), rn.data(), i,
                      1, 1e-12, .804, .066725, .21951, 1.0, 1.0);
    assert(e == ref && er == rr && en == rn);
    std::vector<double> g0(n), g1(n), g2(n), norms(n), saved0(n);
    for (size_t i = 0; i < n; i++) {
      g0[i] = saved0[i] = sin(.13*i); g1[i] = .3*cos(.27*i); g2[i] = -.7*sin(.39*i);
    }
    pw_gpu_launch_norm3(g0.data(), g1.data(), g2.data(), norms.data(), n, 0);
    for (size_t i = 0; i < n; i++)
      assert(norms[i] == sqrt(g0[i]*g0[i] + g1[i]*g1[i] + g2[i]*g2[i]));
    pw_gpu_launch_combine_div_input(g0.data(), g1.data(), g2.data(), en.data(), norms.data(), n, .05, 0);
    for (size_t i = 0; i < n; i++) {
      const double factor = en[i]/fmax(norms[i], .05);
      assert(g0[i] == -(saved0[i]*factor));
      assert(g1[i] == -(.3*cos(.27*i)*factor));
      assert(g2[i] == -(-.7*sin(.39*i)*factor));
    }
    pw_gpu_launch_add_vrho(g0.data(), er.data(), n, 0);
    for (size_t i = 0; i < n; i++)
      assert(g0[i] == er[i] - saved0[i]*(en[i]/fmax(norms[i], .05)));
    for (size_t i = 0; i < n; i++) { rho[i] = (i % 11)*.1; e[i] = 2.; }
    pw_gpu_launch_xc_cutoff(e.data(), rho.data(), n, .25, 3., 0);
    for (size_t i = 0; i < n; i++) {
      double expected_cut = 2.;
      if (rho[i] < .25) expected_cut = 0.;
      else if (rho[i] < .5) {
        double t = (rho[i] - .25)/.25;
        expected_cut = 2.*t*t*(t - .5*t*t);
      } else if (rho[i] < .75) {
        double t = 2. - (rho[i] - .25)/.25;
        expected_cut = 2.*(1. - t*t*(t - .5*t*t));
      }
      assert(fabs(e[i] - expected_cut) < 1e-15);
    }
    const size_t parts = pw_gpu_xc_reduce_size(n);
    std::vector<double> sums(parts);
    pw_gpu_launch_xc_reduce(e.data(), sums.data(), n, 0);
    long double total = 0., partial_total = 0.;
    for (double v : e) total += v;
    for (double v : sums) partial_total += v;
    assert(fabsl(total - partial_total) < 1e-12L*(1 + fabsl(total)));
    std::vector<double> g(3*n, .2), gsq(n, .12);
    double r[6] = {1., 2., 3., -.3, .7, .9}, radii[2] = {.3, .7};
    double cv[4] = {.7, -.2, -.7, .3};
    expected = x;
    pw_gpu_launch_modify_hartree(y.data(), g.data(), gsq.data(), r, radii, cv,
                                 n, 2, 2, 1., 10., 0);
    for (size_t i = 0; i < n; i++)
      pw_hartree_elem(expected.data(), g.data(), gsq.data(), r, radii, cv, i, 2, 2, 1., 10.);
    assert(y == expected);
  }
  // More than two reduction levels, mixed magnitudes, cancellation, and a tail.
  std::vector<double> reduction(70001), partial(548), reduced(6), scalar(1);
  for (size_t i = 0; i < reduction.size(); i++) {
    reduction[i] = i%3 == 0 ? 1e5 : i%3 == 1 ? -.99999999e5 : 1e-7;
  }
  const long double reference = 23334*((long double)reduction[0] + reduction[1]) +
                                23333*(long double)reduction[2];
  pw_gpu_launch_xc_reduce(reduction.data(), partial.data(), reduction.size(), 0);
  pw_gpu_launch_xc_reduce(partial.data(), reduced.data(), partial.size(), 0);
  pw_gpu_launch_xc_reduce(reduced.data(), scalar.data(), reduced.size(), 0);
  if (!(fabsl(reference - scalar[0]) < 1e-12L*(1 + fabsl(reference))))
    std::fprintf(stderr, "reduction: %.21Lg != %.17g (difference %.21Lg)\n", reference, scalar[0], reference - scalar[0]);
  assert(fabsl(reference - scalar[0]) < 1e-12L*(1 + fabsl(reference)));
  puts("PASS production .cu bodies/launchers (CPU launch emulation, including tail blocks and grid strides)");
}
