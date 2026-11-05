// Residency kernels in the emulated .cu translation unit (appended after
// Hermes's kernel body test, whose main is renamed to hermes_main).
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
  std::puts("residency kernels: PASS");
  return 0;
}
