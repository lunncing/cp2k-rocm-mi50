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
    std::vector<double> g(3*n, .2), g2(n, .12);
    double r[6] = {1., 2., 3., -.3, .7, .9}, radii[2] = {.3, .7};
    double cv[4] = {.7, -.2, -.7, .3};
    expected = x;
    pw_gpu_launch_modify_hartree(y.data(), g.data(), g2.data(), r, radii, cv,
                                 n, 2, 2, 1., 10., 0);
    for (size_t i = 0; i < n; i++)
      pw_hartree_elem(expected.data(), g.data(), g2.data(), r, radii, cv, i, 2, 2, 1., 10.);
    assert(y == expected);
  }
  puts("PASS production .cu bodies/launchers (CPU launch emulation, including tail blocks and grid strides)");
}
