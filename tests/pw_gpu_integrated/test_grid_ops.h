/* SPDX-License-Identifier: GPL-2.0-or-later */
static void grid_ops(void) {
  double x[66], y[66], expect[66];
  const int map[] = {13, 2, 8, 1, 9, 16, 3};
  const double scales[] = {0.0, 1.0, -1.3, 0x1.0000000000001p0};
  start(0);
  for (int cx = 1; cx <= 2; cx++) for (int cy = 1; cy <= 2; cy++)
    for (int mapping = 0; mapping <= 2; mapping++) {
      size_t nx = mapping == 2 ? 7 : 17, ny = mapping == 1 ? 7 : 17;
      for (size_t a = 0; a < 4; a++) for (size_t b = 0; b < 4; b++) {
        for (int j = 0; j < 66; j++) {
          x[j] = sin(j * 0.71);
          expect[j] = y[j] = cos(j * 0.27);
        }
        for (size_t j = 0; j < ny * cy; j++)
          expect[j] = scales[b] == 0.0 ? 0.0 : scales[b] == 1.0
              ? expect[j] : expect[j] * scales[b];
        for (size_t i = 0; i < (nx < ny ? nx : ny); i++) {
          size_t src = mapping == 1 ? (size_t)map[i] - 1 : i;
          size_t dst = mapping == 2 ? (size_t)map[i] - 1 : i;
          for (int c = 0; c < cy; c++) {
            double v = c < cx ? x[src * cx + c] : 0.0;
            if (scales[a] == 1.0) expect[dst * cy + c] += v;
            else if (scales[a] != 0.0) expect[dst * cy + c] += scales[a] * v;
          }
        }
        int ok = mapping ? pw_gpu_remap_axpy_c(x, y, nx, ny, cx, cy, map,
                         mapping == 2, scales[a], scales[b])
                         : pw_gpu_axpy_c(x, y, nx, ny, cx, cy, scales[a], scales[b]);
        assert(ok && memcmp(y, expect, ny * cy * sizeof(double)) == 0);
        memset(expect, 0, sizeof(expect));
        for (size_t i = 0; i < (nx < ny ? nx : ny); i++) {
          size_t src = mapping == 1 ? (size_t)map[i] - 1 : i;
          size_t dst = mapping == 2 ? (size_t)map[i] - 1 : i;
          for (int c = 0; c < cy; c++)
            expect[dst * cy + c] = c < cx ? x[src * cx + c] : 0.0;
        }
        ok = mapping ? pw_gpu_remap_copy_c(x, y, nx, ny, cx, cy, map, mapping == 2, 1.0)
                     : pw_gpu_copy_c(x, y, nx, ny, cx, cy, 1.0);
        assert(ok && memcmp(y, expect, ny * cy * sizeof(double)) == 0);
        mock_assert_idle();
        mock_reset_trace();
      }
    }
  x[0] = NAN; y[0] = NAN;
  assert(pw_gpu_axpy_c(x, y, 1, 1, 1, 1, 0.0, 0.0) && y[0] == 0.0);
  x[0] = -0.0;
  assert(pw_gpu_copy_c(x, y, 1, 1, 1, 1, 1.0) && signbit(y[0]));
  assert(pw_gpu_axpy_c(x, x, 1, 1, 1, 1, 1.0, 0.0) == 0 && signbit(x[0]));
  assert(pw_gpu_copy_c(NULL, NULL, 0, 0, 1, 1, 1.0));
  assert(pw_gpu_axpy_c(NULL, NULL, 0, 0, 1, 1, 1.0, 1.0));
  assert(modify_hartree_pot_gpu_c(NULL, NULL, NULL, NULL, NULL, NULL, 0, 0, 0, 1.0, 1.0));
  assert(pbe_lda_eval_gpu_c(NULL, NULL, NULL, NULL, NULL, 0, 0, 1e-12, .804, .066725, .2, 1, 1));
  assert(!pbe_lda_eval_gpu_c(NULL, NULL, NULL, NULL, NULL, 5, 2, 1e-12, .804, .066725, .2, 1, 1));
  finish();
}

static void new_limits(int disabled) {
  double x[18], y[18], saved[18], g[27], gsq[9], r[3] = {1, 2, 3}, radius = .3, cv = .1;
  int map[9];
  for (int j = 0; j < 18; j++) x[j] = y[j] = saved[j] = 1.0 + .1 * j;
  for (int j = 0; j < 27; j++) g[j] = .01 * j;
  for (int j = 0; j < 9; j++) { gsq[j] = .1 + .1 * j; map[j] = 9 - j; }
  // Each operation gets a fresh exact byte budget, then a one-byte-short budget.
  const size_t required[] = {36 * sizeof(double), 36 * sizeof(double),
      36 * sizeof(double) + sizeof(map), 36 * sizeof(double) + sizeof(map),
      59 * sizeof(double), 45 * sizeof(double)};
  for (int op = 0; op < 6; op++) for (int shortfall = 0; shortfall <= 1; shortfall++) {
    if (disabled && op != disabled - 1) continue;
    start(0);
    max_gpu_memory_bytes = mock_limit = required[op] - shortfall;
    pw_gpu_set_host_fallback(!disabled);
    memcpy(y, saved, sizeof(y));
    int ok;
    switch (op) {
    case 0: ok = pw_gpu_copy_c(x, y, 9, 9, 2, 2, 1.0); break;
    case 1: ok = pw_gpu_axpy_c(x, y, 9, 9, 2, 2, .7, .2); break;
    case 2: ok = pw_gpu_remap_copy_c(x, y, 9, 9, 2, 2, map, 0, 1.0); break;
    case 3: ok = pw_gpu_remap_axpy_c(x, y, 9, 9, 2, 2, map, 1, .7, .2); break;
    case 4: ok = modify_hartree_pot_gpu_c(y, g, gsq, r, &radius, &cv, 9, 1, 1, 1, 10); break;
    default: ok = pbe_lda_eval_gpu_c(x, x, y, y + 9, g, 9, 1, 1e-12, .804, .066725, .2, 1, 1);
    }
    assert(ok == !shortfall);
    if (shortfall) assert(memcmp(y, saved, sizeof(y)) == 0 && mock_allocations == 0);
    else assert(mock_allocated == required[op]);
    mock_assert_idle();
    finish();
  }
  start(0);
  assert(!pw_gpu_copy_c(x, y, SIZE_MAX, 1, 2, 2, 1.0));
  assert(!pw_gpu_remap_copy_c(x, y, SIZE_MAX, SIZE_MAX, 2, 2, map, 0, 1.0));
  assert(!pbe_lda_eval_gpu_c(x, x, y, y, y, SIZE_MAX, 1, 0, .804, .066725, .2, 1, 1));
  assert(!modify_hartree_pot_gpu_c(y, g, gsq, r, &radius, &cv, 1, SIZE_MAX, 2, 1, 10));
  assert(!modify_hartree_pot_gpu_c(y, g, gsq, r, &radius, &cv, SIZE_MAX, 1, 1, 1, 10));
  assert(mock_allocations == 0);
  finish();
}
