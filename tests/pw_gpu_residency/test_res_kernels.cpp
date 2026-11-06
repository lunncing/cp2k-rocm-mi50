// Residency kernels in the emulated .cu translation unit (appended after
// Hermes's kernel body test, whose main is renamed to hermes_main).
#include <cmath>
#include <complex>

int main() {
  hermes_main();
  for (size_t n : {size_t(0), size_t(1), size_t(255), size_t(257), size_t(20000)}) {
    std::vector<double> a(2 * n), w1(n), w2(2 * n), out(2 * n, -1), ref(2 * n);
    for (size_t i = 0; i < 2 * n; i++) { a[i] = sin(.31 * i); w2[i] = cos(.17 * i); }
    for (size_t i = 0; i < n; i++) w1[i] = 1.0 / (1.0 + i);
    for (int width = 1; width <= 2; width++) {
      const double *w = width == 1 ? w1.data() : w2.data();
      pw_gpu_launch_res_cmul(out.data(), a.data(), w, width, n, 0);
      for (size_t i = 0; i < n; i++) pw_res_cmul_elem(ref.data(), a.data(), w, width, i);
      assert(out == ref);
    }
    // In place (out aliases a).
    std::vector<double> b = a;
    pw_gpu_launch_res_cmul(b.data(), b.data(), w1.data(), 1, n, 0);
    for (size_t i = 0; i < n; i++) pw_res_cmul_elem(ref.data(), a.data(), w1.data(), 1, i);
    assert(b == ref);
    std::vector<double> p(n + 1, 0);
    pw_gpu_launch_res_cdot_terms(a.data(), w2.data(), p.data(), n, 0);
    for (size_t i = 0; i < n; i++) assert(p[i] == a[2 * i] * w2[2 * i] + a[2 * i + 1] * w2[2 * i + 1]);
    for (int width = 1; width <= 2; width++) {
      const size_t count = pw_res_sample_count(n);
      std::vector<double> s(width * count + 1, -3);
      pw_gpu_launch_res_sample(a.data(), s.data(), n, width, 0);
      for (size_t k = 0; k < count; k++)
        for (int c = 0; c < width; c++)
          assert(s[width * k + c] == a[width * pw_res_sample_index(k, n) + c]);
      assert(s[width * count] == -3);
    }
  }
  // Stage 4: natural-order XC kernels against independent reference loops.
  for (int dims_case = 0; dims_case < 2; dims_case++) {
    const size_t n0 = dims_case ? 7 : 4, n1 = dims_case ? 5 : 3, n2 = dims_case ? 9 : 5;
    const size_t n = n0 * n1 * n2;
    std::vector<int> map(n);
    for (size_t i = 0; i < n; i++) map[i] = (int)i;
    for (size_t i = n - 1; i > 0; i--) std::swap(map[i], map[(i * 7919 + 13) % (i + 1)]);
    std::vector<double> tx(n0), ty(n1), tz(n2), g(3 * n), gfull(3 * n);
    for (size_t i = 0; i < n0; i++) tx[i] = 0.25 * i - 0.5;
    for (size_t i = 0; i < n1; i++) ty[i] = -0.375 * i + 0.125;
    for (size_t i = 0; i < n2; i++) tz[i] = 0.0625 * i * i - 1.0;
    tx[0] = -0.0; // signed zeros must be preserved bitwise
    for (size_t i = 0; i < n; i++) {
      const size_t p = (size_t)map[i];
      g[3 * i] = tx[p % n0]; g[3 * i + 1] = ty[(p / n0) % n1]; g[3 * i + 2] = tz[p / (n0 * n1)];
      for (int d = 0; d < 3; d++) gfull[3 * p + d] = g[3 * i + d];
    }
    std::vector<double> gt(n0 + n1 + n2, 99.0), flags(n, -1);
    pw_gpu_launch_gtab_build(gt.data(), g.data(), map.data(), n0, n1, n, 0);
    for (size_t i = 0; i < n0; i++) assert(std::memcmp(&gt[i], &tx[i], 8) == 0);
    for (size_t i = 0; i < n1; i++) assert(gt[n0 + i] == ty[i]);
    for (size_t i = 0; i < n2; i++) assert(gt[n0 + n1 + i] == tz[i]);
    pw_gpu_launch_gtab_check(flags.data(), g.data(), map.data(), gt.data(), n0, n1, n, 0);
    for (size_t i = 0; i < n; i++) assert(flags[i] == 0.0);
    std::vector<double> gbad = g;
    gbad[3 * 5 + 1] = std::nextafter(gbad[3 * 5 + 1], 10.0);
    pw_gpu_launch_gtab_check(flags.data(), gbad.data(), map.data(), gt.data(), n0, n1, n, 0);
    double nbad = 0;
    for (double f : flags) nbad += f;
    assert(flags[5] == 1.0 && nbad == 1.0);
    // +0 versus -0 counts as a difference
    std::vector<double> gz = g;
    for (size_t i = 0; i < n; i++) if ((size_t)map[i] % n0 == 0) { gz[3 * i] = 0.0; break; }
    pw_gpu_launch_gtab_check(flags.data(), gz.data(), map.data(), gt.data(), n0, n1, n, 0);
    nbad = 0;
    for (double f : flags) nbad += f;
    assert(nbad == 1.0);
    std::vector<double> gt2(3 * n + 1, -4.0);
    pw_gpu_launch_gtab_full(gt2.data(), g.data(), map.data(), n, 0);
    assert(std::memcmp(gt2.data(), gfull.data(), 24 * n) == 0 && gt2[3 * n] == -4.0);
    std::vector<double> z(2 * n), c(2 * n + 1, -9), cref(2 * n), acc(2 * n), accref(2 * n);
    for (size_t i = 0; i < 2 * n; i++) { z[i] = std::sin(0.7 * i + 0.1); acc[i] = accref[i] = std::cos(0.3 * i); }
    for (int mode = 1; mode <= 2; mode++) {
      const double *t = mode == 1 ? gt.data() : gfull.data();
      for (int d = 0; d < 3; d++) {
        const double scale = d == 1 ? 1.0 : 0.37;
        pw_gpu_launch_nat_deriv(c.data(), z.data(), t, mode, n0, n1, d, scale, n, 0);
        pw_gpu_launch_nat_deriv_acc(acc.data(), z.data(), t, mode, n0, n1, d, scale, n, 0);
        for (size_t p = 0; p < n; p++) {
          const double gd = gfull[3 * p + d];
          cref[2 * p] = scale * -(z[2 * p + 1] * gd);
          cref[2 * p + 1] = scale * (z[2 * p] * gd);
          accref[2 * p] = accref[2 * p] + -((scale * z[2 * p + 1]) * gd);
          accref[2 * p + 1] = accref[2 * p + 1] + (scale * z[2 * p]) * gd;
        }
        assert(std::memcmp(c.data(), cref.data(), 16 * n) == 0 && c[2 * n] == -9);
        assert(std::memcmp(acc.data(), accref.data(), 16 * n) == 0);
      }
    }
  }
  // Stage 5: paired XC kernels and the fused Hartree begin.
  {
    auto close = [](double a, double b) { return std::fabs(a - b) <= 4e-16 * (1.0 + std::fabs(b)); };
    for (int dims_case = 0; dims_case < 2; dims_case++) {
      const size_t n0 = dims_case ? 6 : 4, n1 = dims_case ? 4 : 3, n2 = dims_case ? 2 : 5;
      const size_t n = n0 * n1 * n2;
      // mirror index against the coordinates
      for (size_t p = 0; p < n; p++) {
        const size_t x = p % n0, y = (p / n0) % n1, z = p / (n0 * n1);
        const size_t m = (n0 - x) % n0 + n0 * ((n1 - y) % n1 + n1 * ((n2 - z) % n2));
        assert(pw_nat_mirror(p, n0, n1, n2) == m && pw_nat_mirror(m, n0, n1, n2) == p);
      }
      std::vector<double> gt(3 * n), rho(2 * n), z(2 * n + 1, -9), w(2 * n), acc(2 * n);
      for (size_t i = 0; i < 3 * n; i++) gt[i] = std::sin(1.3 * i + 0.2);
      for (size_t i = 0; i < 2 * n; i++) { rho[i] = std::cos(0.37 * i); w[i] = std::sin(0.11 * i + 1.0); }
      auto gd = [&](size_t q, int d) { return gt[3 * q + d]; };
      typedef std::complex<double> cd;
      for (int a = 0; a < 3; a++) {
        const int b = (a + 1) % 3;
        pw_gpu_launch_nat_grad2(z.data(), rho.data(), gt.data(), 2, n0, n1, n2, a, b, n, 0);
        for (size_t p = 0; p < n; p++) {
          const size_t m = pw_nat_mirror(p, n0, n1, n2);
          const cd r(rho[2 * p], rho[2 * p + 1]), rm(rho[2 * m], rho[2 * m + 1]), I(0, 1);
          const cd fa = 0.5 * (I * gd(p, a) * r + std::conj(I * gd(m, a) * rm));
          const cd fb = 0.5 * (I * gd(p, b) * r + std::conj(I * gd(m, b) * rm));
          const cd ref = fa + I * fb;
          assert(close(z[2 * p], ref.real()) && close(z[2 * p + 1], ref.imag()));
        }
        assert(z[2 * n] == -9);
        for (int accumulate = 0; accumulate <= 1; accumulate++) {
          for (size_t i = 0; i < 2 * n; i++) acc[i] = std::cos(0.5 * i);
          const std::vector<double> acc0 = acc;
          pw_gpu_launch_nat_div2(acc.data(), w.data(), gt.data(), 2, n0, n1, n2, a, b, 0.25,
                                 accumulate, n, 0);
          for (size_t p = 0; p < n; p++) {
            const size_t m = pw_nat_mirror(p, n0, n1, n2);
            const cd wp(w[2 * p], w[2 * p + 1]), wm(w[2 * m], w[2 * m + 1]), I(0, 1);
            const cd ha = 0.5 * (wp + std::conj(wm)), hb = (wp - std::conj(wm)) / (2.0 * I);
            const cd v = I * 0.25 * (gd(p, a) * ha + gd(p, b) * hb);
            const cd ref = accumulate ? cd(acc0[2 * p], acc0[2 * p + 1]) + v : v;
            assert(close(acc[2 * p], ref.real()) && close(acc[2 * p + 1], ref.imag()));
          }
        }
      }
      // Separation of two real fields: div2 of FFT-like W built from real h_a,
      // h_b recovers each spectrum (here W(p) = Ha + i Hb with Hermitian H).
      std::vector<double> ha(2 * n), hb(2 * n), ww(2 * n), out(2 * n), refo(2 * n);
      for (size_t p = 0; p < n; p++) {
        const size_t m = pw_nat_mirror(p, n0, n1, n2);
        if (m < p) continue;
        const double ar = std::sin(0.3 * p), ai = m == p ? 0.0 : std::cos(0.7 * p);
        const double br = std::cos(0.2 * p + 1), bi = m == p ? 0.0 : std::sin(0.9 * p);
        ha[2 * p] = ar; ha[2 * p + 1] = ai; ha[2 * m] = ar; ha[2 * m + 1] = -ai;
        hb[2 * p] = br; hb[2 * p + 1] = bi; hb[2 * m] = br; hb[2 * m + 1] = -bi;
      }
      for (size_t p = 0; p < n; p++) {
        ww[2 * p] = ha[2 * p] - hb[2 * p + 1];
        ww[2 * p + 1] = ha[2 * p + 1] + hb[2 * p];
      }
      pw_gpu_launch_nat_div2(out.data(), ww.data(), gt.data(), 2, n0, n1, n2, 0, 1, 1.0, 0, n, 0);
      for (size_t p = 0; p < n; p++) {
        const cd I(0, 1);
        const cd v = I * (gd(p, 0) * cd(ha[2 * p], ha[2 * p + 1]) + gd(p, 1) * cd(hb[2 * p], hb[2 * p + 1]));
        assert(close(out[2 * p], v.real()) && close(out[2 * p + 1], v.imag()));
      }
      std::vector<double> x(n), y(n), c(2 * n), x2(n), y2(n);
      for (size_t p = 0; p < n; p++) { x[p] = std::sin(0.4 * p); y[p] = -std::cos(0.8 * p); }
      pw_gpu_launch_pack2(x.data(), y.data(), c.data(), n, 0);
      pw_gpu_launch_unpack2(c.data(), x2.data(), y2.data(), n, 0);
      for (size_t p = 0; p < n; p++)
        assert(c[2 * p] == x[p] && c[2 * p + 1] == y[p] && x2[p] == x[p] && y2[p] == y[p]);
      // Fused Hartree begin: bitwise copy + pw_axpy + pw_res_cmul_elem, also in place.
      for (int width = 1; width <= 2; width++) {
        std::vector<double> core(2 * n), rg(2 * n), wf(2 * n), tot(2 * n), vh(2 * n);
        std::vector<double> rtot(2 * n), rvh(2 * n);
        for (size_t i = 0; i < 2 * n; i++) { core[i] = std::sin(0.9 * i); rg[i] = std::cos(1.7 * i); wf[i] = 1.0 / (i + 1.5); }
        for (size_t i = 0; i < 2 * n; i++) rtot[i] = core[i] + rg[i];
        for (size_t i = 0; i < n; i++) pw_res_cmul_elem(rvh.data(), rtot.data(), wf.data(), width, i);
        pw_gpu_launch_res_tot_cmul(tot.data(), vh.data(), core.data(), rg.data(), wf.data(), width, n, 0);
        assert(std::memcmp(tot.data(), rtot.data(), 16 * n) == 0 && std::memcmp(vh.data(), rvh.data(), 16 * n) == 0);
        // core uploaded into rho_tot, rho_g uploaded into v_H
        std::vector<double> t2 = core, v2 = rg;
        pw_gpu_launch_res_tot_cmul(t2.data(), v2.data(), t2.data(), v2.data(), wf.data(), width, n, 0);
        assert(std::memcmp(t2.data(), rtot.data(), 16 * n) == 0 && std::memcmp(v2.data(), rvh.data(), 16 * n) == 0);
      }
    }
  }
  // Stage 6: rho_r and d rho/dz from one inverse transform, for a rho without
  // Hermitian symmetry, odd and even axes (Nyquist planes), both g layouts.
  {
    typedef std::complex<double> cd;
    const double pi = std::acos(-1.0);
    for (int dims_case = 0; dims_case < 2; dims_case++) {
      const size_t n0 = dims_case ? 6 : 5, n1 = dims_case ? 4 : 3, n2 = dims_case ? 2 : 7;
      const size_t n = n0 * n1 * n2;
      for (int mode = 1; mode <= 2; mode++) {
        std::vector<double> gt(mode == 1 ? n0 + n1 + n2 : 3 * n), rho(2 * n), z(2 * n + 1, -9);
        for (size_t i = 0; i < gt.size(); i++) gt[i] = std::sin(0.9 * i + 0.4);
        for (size_t i = 0; i < 2 * n; i++) rho[i] = std::cos(0.43 * i) + 0.2 * std::sin(1.9 * i);
        pw_gpu_launch_nat_rho_dz(z.data(), rho.data(), gt.data(), mode, n0, n1, n2, n, 0);
        assert(z[2 * n] == -9);
        const cd I(0, 1);
        auto gz = [&](size_t q) { return pw_nat_gd(gt.data(), mode, n0, n1, q, 2); };
        for (size_t p = 0; p < n; p++) {
          const size_t m = pw_nat_mirror(p, n0, n1, n2);
          const cd r(rho[2 * p], rho[2 * p + 1]), rm(rho[2 * m], rho[2 * m + 1]);
          const cd ref = 0.5 * (r + std::conj(rm)) +
                         I * (0.5 * (I * gz(p) * r + std::conj(I * gz(m) * rm)));
          assert(std::fabs(z[2 * p] - ref.real()) <= 4e-16 * (1.0 + std::fabs(ref.real())));
          assert(std::fabs(z[2 * p + 1] - ref.imag()) <= 4e-16 * (1.0 + std::fabs(ref.imag())));
        }
        // Direct inverse DFT: Re/Im of IFFT[z] are Re IFFT[rho] and Re IFFT[i g_z rho].
        double worst = 0.0;
        for (size_t x = 0; x < n; x++) {
          const size_t x0 = x % n0, x1 = (x / n0) % n1, x2 = x / (n0 * n1);
          cd sz = 0, sr = 0, sd = 0;
          for (size_t p = 0; p < n; p++) {
            const size_t k0 = p % n0, k1 = (p / n0) % n1, k2 = p / (n0 * n1);
            const double ph = 2 * pi * ((double)(k0 * x0) / n0 + (double)(k1 * x1) / n1 +
                                        (double)(k2 * x2) / n2);
            const cd e(std::cos(ph), std::sin(ph)), r(rho[2 * p], rho[2 * p + 1]);
            sz += cd(z[2 * p], z[2 * p + 1]) * e;
            sr += r * e;
            sd += I * gz(p) * r * e;
          }
          worst = std::fmax(worst, std::fabs(sz.real() - sr.real()));
          worst = std::fmax(worst, std::fabs(sz.imag() - sd.real()));
        }
        assert(worst <= 1e-12);
      }
    }
  }
  std::puts("residency kernels: PASS");
  return 0;
}
