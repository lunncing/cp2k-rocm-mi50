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
    for (int derivative : {-1, 0, 1}) {
      std::vector<double> ra(n), rb(n), lsd[6], oracle[6];
      for (size_t point = 0; point < n; point++) {
        ra[point] = point % 7 == 0 ? -0.1 : 0.3 + .01 * (point % 9);
        rb[point] = point % 5 == 0 ? 0.0 : 0.2;
      }
      if (n > 0) { ra[0] = 0.0; rb[0] = 1.0; }
      if (n > 1) { ra[1] = 1.0; rb[1] = 0.0; }
      for (int output = 0; output < 6; output++) {
        lsd[output].assign(n + 1, .017);
        oracle[output] = lsd[output];
      }
      pw_gpu_launch_pbe_lsd(ra.data(), rb.data(), norm.data(), norm.data(), norm.data(),
          lsd[0].data(), lsd[1].data(), lsd[2].data(), lsd[3].data(), lsd[4].data(),
          lsd[5].data(), n, derivative, 1e-12, .9, .7, 0);
      for (size_t point = 0; point < n; point++)
        pw_pbe_lsd_elem(ra[point], rb[point], norm[point], norm[point], norm[point],
            1e-12, .9, .7, derivative, &oracle[0][point], &oracle[1][point],
            &oracle[2][point], &oracle[3][point], &oracle[4][point], &oracle[5][point]);
      for (int output = 0; output < 6; output++) {
        assert(lsd[output] == oracle[output]);
        assert(lsd[output][n] == .017);
        for (size_t point = 0; point < n; point++) {
          assert(std::isfinite(lsd[output][point]));
          if ((output == 0 && derivative < 0) || (output > 0 && derivative == 0))
            assert(lsd[output][point] == .017);
        }
        lsd[output].assign(n + 1, .017);
      }
      if (n > 0 && derivative != 0) {
        assert(oracle[1][0] != .017 && oracle[2][0] != .017);
        if (n > 1) assert(oracle[1][1] != .017 && oracle[2][1] != .017);
      }
      emulate(pw_pbe_lsd_pointwise, 2, 7, 0, 0, ra.data(), rb.data(), norm.data(),
          norm.data(), norm.data(), derivative >= 0 ? lsd[0].data() : nullptr,
          derivative ? lsd[1].data() : nullptr, derivative ? lsd[2].data() : nullptr,
          derivative ? lsd[3].data() : nullptr, derivative ? lsd[4].data() : nullptr,
          derivative ? lsd[5].data() : nullptr, n, derivative, 1e-12, .9, .7);
      for (int output = 0; output < 6; output++) assert(lsd[output] == oracle[output]);
    }
    {
      std::vector<double> gradients[6], original[6], ra(n, .3), rb(n, .7);
      std::vector<double> va(n + 1, -7), vb(n + 1, -7), energy(n + 1, -7);
      for (int dim = 0; dim < 6; dim++) {
        gradients[dim].resize(n + 1, -7);
        for (size_t point = 0; point < n; point++)
          gradients[dim][point] = .1 * sin(.03 * point + dim);
        original[dim] = gradients[dim];
      }
      pw_gpu_launch_xc_pbe_lsd_combine(ra.data(), rb.data(), gradients[0].data(),
          gradients[1].data(), gradients[2].data(), gradients[3].data(), gradients[4].data(),
          gradients[5].data(), va.data(), vb.data(), energy.data(), n,
          1e-12, .9, .7, .02, 0.0, 0);
      for (size_t point = 0; point < n; point++) {
        double sum[3];
        for (int dim = 0; dim < 3; dim++) sum[dim] = original[dim][point] + original[dim + 3][point];
        const double norm_a = pw_xc_norm3(original[0][point], original[1][point], original[2][point]);
        const double norm_b = pw_xc_norm3(original[3][point], original[4][point], original[5][point]);
        const double norm_total = pw_xc_norm3(sum[0], sum[1], sum[2]);
        double e0 = 0, era = 0, erb = 0, en = 0, ena = 0, enb = 0;
        pw_pbe_lsd_elem(ra[point], rb[point], norm_total, norm_a, norm_b,
                        1e-12, .9, .7, 1, &e0, &era, &erb, &en, &ena, &enb);
        for (int dim = 0; dim < 3; dim++) {
          const double shared = sum[dim] * (en / fmax(norm_total, .02));
          assert(gradients[dim][point] == -((ena / fmax(norm_a, .02)) * original[dim][point] + shared));
          assert(gradients[dim + 3][point] == -((enb / fmax(norm_b, .02)) * original[dim + 3][point] + shared));
        }
        assert(va[point] == era && vb[point] == erb && energy[point] == e0);
      }
      assert(va[n] == -7 && vb[n] == -7 && energy[n] == -7);
      for (int dim = 0; dim < 6; dim++) assert(gradients[dim][n] == -7);
    }
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
    std::vector<double> ra(n), rb(n), lsd_cutoff(n + 1, 2.);
    for (size_t point = 0; point < n; point++) {
      ra[point] = .25 * rho[point];
      rb[point] = rho[point] - ra[point];
      rho[point] = ra[point] + rb[point];
    }
    pw_gpu_launch_xc_cutoff_lsd(lsd_cutoff.data(), ra.data(), rb.data(), n, .25, 0., 0);
    for (double value : lsd_cutoff) assert(value == 2.);
    pw_gpu_launch_xc_cutoff_lsd(lsd_cutoff.data(), ra.data(), rb.data(), n, .25, 3., 0);
    assert(lsd_cutoff[n] == 2.);
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
      assert(lsd_cutoff[i] == e[i]);
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
