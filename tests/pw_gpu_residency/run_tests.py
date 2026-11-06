#!/usr/bin/env python3
"""PW GPU residency tests against the checked CPU mock (no GPU required).

Builds the production src/pw/gpu C sources with ASan/UBSan and runs
  * Hermes's C cases from ../pw_gpu_integrated (must keep passing),
  * the emulated .cu kernel bodies (Hermes's checks + the residency kernels),
  * the residency cases: every fused path bit for bit against the original
    call sequence through the same mock FFT, with 1 and 3 OpenMP threads.
Usage (from the CP2K source root):  python3 tests/pw_gpu_residency/run_tests.py
"""
import os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HERMES = HERE.parent / "pw_gpu_integrated"
if not (HERMES / "mock_offload.c").is_file():
    sys.exit(f"missing {HERMES} (Hermes's pw_gpu_integrated test kit, updated version)")
sys.path.insert(0, str(HERMES))
sys.dont_write_bytecode = True
import run_tests as hermes  # noqa: E402
from fortran_tests import run_fortran_tests  # noqa: E402
from grid_tests import run_grid_tests  # noqa: E402

RES_CASES = ["density", "density_rejects", "potential", "hartree", "chain", "memory",
             "borrowed", "samples", "verify"]

# Extend the checked mock only in the temporary build. The production .cu
# bodies are independently emulated below; these adapters preserve the mock's
# bounds checks and asynchronous stream dependency checks in every C/Fortran
# executable, including the existing integrated tests.
NATURAL_MOCK = r'''
#include "pw_gpu_kernels.h"
#include <assert.h>
#include <limits.h>
size_t mock_natural_mul_n, mock_natural_dot_n, mock_natural_prefix_n;
void pw_gpu_launch_res_natural_cmul(double *v, const double *rho,
    const double *infl, int width, size_t n, offloadStream_t s) {
  mock_natural_mul_n += n;
  pw_gpu_launch_res_cmul(v, rho, infl, width, n, s);
}
void pw_gpu_launch_res_natural_cdot_terms(const double *rho, const double *v,
    double *terms, size_t n, offloadStream_t s) {
  mock_natural_dot_n += n;
  pw_gpu_launch_res_cdot_terms(rho, v, terms, n, s);
}
void pw_gpu_launch_res_natural_extract_prefix(const double *src, double *out,
    const int *coords, size_t n, offloadStream_t s) {
  assert(n <= INT_MAX);
  mock_natural_prefix_n += n;
  if (n) pw_gpu_launch_gather(out, src, 1.0, (int)n, coords, s);
}
'''

NATURAL_KERNEL_TESTS = r'''
static void natural_kernel_tests() {
  // Independent CPU oracle from the verified natural_fft_cpu.c prototype.
  for (size_t n : {size_t(0), size_t(1), size_t(255), size_t(257), size_t(20000)}) {
    std::vector<double> rho(2 * n), infl(2 * n), real(n), out(2 * n + 1, -9);
    for (size_t i = 0; i < 2 * n; i++) { rho[i] = sin(.13 * i); infl[i] = cos(.27 * i); }
    for (size_t i = 0; i < n; i++) real[i] = infl[2 * i];
    for (int width : {1, 2}) {
      const double *w = width == 1 ? real.data() : infl.data();
      pw_gpu_launch_res_natural_cmul(out.data(), rho.data(), w, width, n, 0);
      std::vector<double> alias = rho;
      pw_gpu_launch_res_natural_cmul(alias.data(), alias.data(), w, width, n, 0);
      for (size_t i = 0; i < n; i++) {
        const double ar = rho[2 * i], ai = rho[2 * i + 1];
        const double br = w[width * i], bi = width == 2 ? w[2 * i + 1] : 0.0;
        assert(out[2 * i] == ar * br - ai * bi);
        assert(out[2 * i + 1] == ar * bi + ai * br);
        assert(alias[2 * i] == out[2 * i] && alias[2 * i + 1] == out[2 * i + 1]);
      }
      assert(out[2 * n] == -9);
    }
    std::vector<double> terms(n + 1, -9);
    pw_gpu_launch_res_natural_cdot_terms(rho.data(), infl.data(), terms.data(), n, 0);
    for (size_t i = 0; i < n; i++)
      assert(terms[i] == rho[2 * i] * infl[2 * i] + rho[2 * i + 1] * infl[2 * i + 1]);
    assert(terms[n] == -9);
    std::vector<int> coords(n);
    for (size_t i = 0; i < n; i++) coords[i] = (int)((i * 17 + n - 1) % n);
    pw_gpu_launch_res_natural_extract_prefix(rho.data(), out.data(), coords.data(), n, 0);
    for (size_t i = 0; i < n; i++) {
      assert(out[2 * i] == rho[2 * coords[i]]);
      assert(out[2 * i + 1] == rho[2 * coords[i] + 1]);
    }
    assert(out[2 * n] == -9);
  }
}
'''

NATURAL_CASES = r'''
#define main residency_main
#include "test_res.c"
#undef main

// Stage 5 pairs two real fields per FFT in the natural XC: the XC outputs then
// agree with the sorted order to rounding instead of bitwise.
static void xc_same(const double *a, const double *b, const size_t n, const char *what) {
  if (!opt_res_xc_pairs) {
    same(a, b, n, what);
    return;
  }
  double big = 0.0;
  for (size_t i = 0; i < n; i++) big = fmax(big, fabs(b[i]));
  for (size_t i = 0; i < n; i++) {
    if (!(fabs(a[i] - b[i]) <= 1e-12 * (1.0 + big))) {
      fprintf(stderr, "%s: mismatch at %zu: %.17g != %.17g\n", what, i, a[i], b[i]);
      abort();
    }
  }
}

static void full_grid(void) {
  grids(1);
  const int n = nr_[0];
  ng_[0] = n;
  free(map_[0]);
  map_[0] = malloc(sizeof(int) * (size_t)n);
  // An anisotropic box, nontrivial ordering, and G=0 away from box index 0.
  for (int i = 0; i < n; i++) map_[0][i] = (7 * i + 17) % n;
  free(g_fine); free(gsq_fine);
  g_fine = rarray(3 * (size_t)n);
  gsq_fine = rarray((size_t)n);
  for (int i = 0; i < n; i++) gsq_fine[i] = (double)i;
}

static void no_target_map(void) {
  for (int i = 0; i < PW_GPU_GRID_CACHE_SIZE; i++) assert(grid_cache[i].grid_id != 1);
}

static void natural_chain(void) {
  full_grid();
  start(0);
  const size_t n = (size_t)ng_[0];
  const bool natural = opt_res_natural_g;
  double *r[NLEV];
  for (int l = 0; l < NLEV; l++) {
    r[l] = rarray((size_t)nr_[l]);
    for (int i = 0; i < nr_[l]; i++) r[l][i] = 0.5 + 0.2 * r[l][i];
  }
  double rg[128] = {0}, rr[64] = {0}, ref_g[128], ref_r[64];
  double tot[128], vh[64], ref_tot[128], ref_vh[64], ref_energy;
  double vxc[64], ref_vxc[64], exc, ref_exc;
  density_reference(r, ref_g, ref_r);
  assert(xc_call(ref_g, ref_r, ref_vxc, &ref_exc));
  pw_gpu_res_ensure_host_c(RES_VXC, ref_vxc);
  hartree_in h = hartree_inputs();
  memcpy(h.rho_g, ref_g, 2 * n * sizeof(double));
  const size_t nprefix = 10;
  for (int pass = 0; pass < 3; pass++) {
    if (pass == 2) { h.core[2] += .25; h.infl[5] = .001; }
    hartree_reference(&h, ref_tot, ref_vh, &ref_energy);
    assert(density_fused(r, rg, rr));
    assert(res[RES_RHO_G].natural_g == natural);
    assert(res_verify_full(&res[RES_RHO_G], ref_g));
    const unsigned long hits = res[RES_RHO_CORE].hits;
    for (size_t i = 0; i < 2 * n; i++) tot[i] = -987.0;
    mock_reset_trace();
    mock_natural_mul_n = mock_natural_dot_n = mock_natural_prefix_n = mock_tot_cmul_n = 0;
    assert(pw_gpu_res_hartree_begin_c(h.core, rg, h.infl, tot, n, nprefix, dims[0]));
    assert(res_hartree.natural_g == natural);
    same(tot, ref_tot, 2 * nprefix, "natural DDAPC prefix");
    for (size_t i = 2 * nprefix; i < 2 * n; i++) assert(tot[i] == -987.0);
    assert(mock_download_bytes == 2 * nprefix * sizeof(double) +
           (opt_res_verify ? (pass == 1 ? 5 : 2) * n * sizeof(double) : 0));
    if (natural) {
      assert(mock_tot_cmul_n == n && mock_natural_prefix_n == nprefix); // fused begin (stage 5)
      assert(res[RES_RHO_CORE].owned_bytes == 2 * n * sizeof(double));
      assert(res[RES_INFL].width == (pass == 2 ? 2 : 1));
    }
    if (pass == 1) assert(res[RES_RHO_CORE].hits == hits + 1);
    assert(pw_gpu_res_hartree_modify_c(h.g0_delta, 1, g_fine + 3, gsq_fine + 1,
        h.r, h.radii, h.cv, 1, n - 1, (size_t)h.natoms, (size_t)h.nradii, h.gcut2, h.vol));
    double dot;
    assert(pw_gpu_res_hartree_energy_c(&dot));
    assert(fabs(0.5 * dot * h.vol - ref_energy) <= 1e-13 * fmax(1.0, fabs(ref_energy)));
    if (natural) assert(mock_natural_dot_n == n);
    mock_reset_trace();
    assert(pw_gpu_res_hartree_finish_c(vh, dims[0], (int)n, 1, 1, map_[0], h.dvol));
    if (natural) {
      for (size_t i = 0; i < mock_operation_count; i++)
        assert(mock_operations[i].kind != MOCK_SCATTER && mock_operations[i].kind != MOCK_GATHER);
    }
    pw_gpu_res_ensure_host_c(RES_V_H_R, vh);
    same(vh, ref_vh, n, "natural Hartree inverse");
    if (pass != 2) {
      pw_gpu_res_ensure_host_c(RES_RHO_G, rg + 7);
      same(rg, ref_g, 2 * n, "natural sorted host sync");
    }
  }
  // XC still consumes sorted coefficients; hand off without uploading rho_g.
  const unsigned long hits = res[RES_RHO_G].hits;
  const unsigned long syncs = res_stats.syncs;
  // The XC wrapper can supply an equal map at a different address. Grid ID
  // and dimensions define the natural layout, not the temporary's address.
  int *saved_map = map_[0];
  map_[0] = malloc(n * sizeof(int));
  memcpy(map_[0], saved_map, n * sizeof(int));
  assert(xc_call(rg, rr, vxc, &exc));
  free(map_[0]);
  map_[0] = saved_map;
  assert(res_stats.syncs == syncs);
  assert(!res[RES_RHO_G].dev_authoritative && !res[RES_RHO_R].dev_authoritative);
  pw_gpu_res_ensure_host_c(RES_VXC, vxc);
  xc_same(vxc, ref_vxc, n, "natural XC handoff");
  xc_same(&exc, &ref_exc, 1, "natural XC energy");
  assert(res[RES_RHO_G].hits == hits + 1);
  // Switching the coarse index cache back to sorted G must not reuse natural indices.
  double *pot[NLEV], *ref_pot[NLEV];
  const double rscale[NLEV] = {1.0, 8.0, 0.5};
  for (int l = 0; l < NLEV; l++) { pot[l] = rarray((size_t)nr_[l]); ref_pot[l] = rarray((size_t)nr_[l]); }
  potential_reference(vxc, ref_pot, rscale);
  assert(potential_fused(vxc, pot, rscale));
  for (int l = 0; l < NLEV; l++) {
    same(pot[l], ref_pot[l], (size_t)nr_[l], "natural to sorted interpolation");
    free(pot[l]); free(ref_pot[l]);
  }
  pw_gpu_res_set_lazy_c(0);
  finish();
  for (int l = 0; l < NLEV; l++) free(r[l]);
  hartree_free(&h);
}

static void natural_lifetime(void) {
  full_grid();
  start(0);
  assert(opt_res_natural_g);
  double *r[NLEV];
  for (int l = 0; l < NLEV; l++) r[l] = rarray((size_t)nr_[l]);
  double rg[128] = {0}, rr[64] = {0}, ref_g[128], ref_r[64], tot[128], vh[64];
  const size_t n = (size_t)ng_[0];
  density_reference(r, ref_g, ref_r);
  hartree_in h = hartree_inputs();
  assert(density_fused(r, rg, rr));
  // Evict a device-only density with no allocation headroom; preserve host order.
  max_gpu_memory_bytes = mock_limit = current_gpu_allocated_bytes;
  release_xc_buffer();
  same(rg, ref_g, 2 * n, "natural eviction G");
  same(rr, ref_r, n, "natural eviction R");
  max_gpu_memory_bytes = mock_limit = 0;
  assert(density_fused(r, rg, rr));
  assert(pw_gpu_res_hartree_begin_c(h.core, rg, h.infl, tot, n, 1, dims[0]));
  pw_gpu_grid_invalidate(1);
  assert(!res_hartree.pending && !res_natural.map);
  same(rg, ref_g, 2 * n, "natural grid invalidation");
  assert(res[RES_RHO_CORE].state == RES_INVALID && res[RES_INFL].state == RES_INVALID);
  for (int i = 0; i < PW_GPU_GRID_CACHE_SIZE; i++) assert(!grid_cache[i].gidx_natural_map);
  // Reordering the same host map after invalidation rebuilds every dependent cache.
  permutation(map_[0], (int)n);
  density_reference(r, ref_g, ref_r);
  assert(density_fused(r, rg, rr));
  assert(res_verify_full(&res[RES_RHO_G], ref_g));
  assert(pw_gpu_res_hartree_begin_c(h.core, rg, h.infl, tot, n, 1, dims[0]));
  const int wrong_dims[3] = {5, 3, 4};
  assert(!pw_gpu_res_hartree_finish_c(vh, wrong_dims, (int)n, 1, 1, map_[0], h.dvol));
  // Switching off the new representation materializes a usable sorted input.
  opt_res_natural_g = false;
  assert(pw_gpu_res_hartree_begin_c(h.core, rg, h.infl, tot, n, 1, dims[0]));
  assert(!res_hartree.natural_g && !res[RES_RHO_CORE].natural_g);
  same(rg, ref_g, 2 * n, "natural switch fallback");
  pw_gpu_res_hartree_cancel_c();
  opt_res_natural_g = true;
  pw_gpu_grid_invalidate(1);
  map_[0][1] = map_[0][0];
  assert(!density_fused(r, rg, rr)); // invalid full-box permutation is rejected
  pw_gpu_res_set_lazy_c(0);
  finish();
  for (int l = 0; l < NLEV; l++) free(r[l]);
  hartree_free(&h);
}

static void natural_memory(void) {
  full_grid();
  start(0);
  const size_t n = (size_t)ng_[0];
  double *r[NLEV];
  for (int l = 0; l < NLEV; l++) r[l] = rarray((size_t)nr_[l]);
  double rg[128] = {0}, rr[64] = {0}, ref_g[128], ref_r[64];
  double tot[128], vh[64], ref_tot[128], ref_vh[64], ref_energy, energy;
  density_reference(r, ref_g, ref_r);
  hartree_in h = hartree_inputs();
  memcpy(h.rho_g, ref_g, 2 * n * sizeof(double));
  hartree_reference(&h, ref_tot, ref_vh, &ref_energy);
  assert(density_fused(r, rg, rr));
  // Model the previous XC call's large target cache. Reclaim it before trying
  // to cache the core; there is exactly enough room for the three owned slots.
  grid_cache_entry *e = grid_cache_lookup_or_upload(1, map_[0], (int)n, 1);
  assert(e != NULL && grid_cache_get_g(e, g_fine) != NULL);
  offloadStreamSynchronize(stream);
  const size_t cache_bytes = n * (sizeof(int) + 3 * sizeof(double));
  max_gpu_memory_bytes = mock_limit = current_gpu_allocated_bytes - cache_bytes +
                                     4 * n * sizeof(double);
  hartree_in device_h = h;
  device_h.rho_g = rg;
  for (int pass = 0; pass < 3; pass++) {
    assert(density_fused(r, rg, rr));
    const unsigned long hits = res[RES_RHO_CORE].hits;
    assert(hartree_fused(&device_h, tot, vh, &energy, 0));
    no_target_map(); // evicted as the least recently used entry to fit the core
    assert(res[RES_RHO_CORE].owned_bytes == 2 * n * sizeof(double));
    assert(current_gpu_allocated_bytes == max_gpu_memory_bytes);
    if (pass) assert(res[RES_RHO_CORE].hits == hits + 1);
    pw_gpu_res_ensure_host_c(RES_V_H_R, vh);
    same(vh, ref_vh, n, "natural exact budget");
  }
  for (int id = RES_V_H_R; id < RES_NSLOTS; id++) res_free_owned(id);
  mock_free_memory_value = res_reserve_bytes;
  assert(density_fused(r, rg, rr));
  assert(hartree_fused(&device_h, tot, vh, &energy, 0));
  same(vh, ref_vh, n, "natural uncached inputs");
  for (int id = RES_V_H_R; id < RES_NSLOTS; id++) assert(!res[id].owned_bytes);
  mock_free_memory_value = (size_t)1 << 40;
  pw_gpu_res_set_lazy_c(0);
  finish();
  max_gpu_memory_bytes = mock_limit = 0;
  for (int l = 0; l < NLEV; l++) free(r[l]);
  hartree_free(&h);
}

static void natural_chunked_sync(void) {
  start(0);
  const size_t n = PW_GPU_RES_NATURAL_CHUNK + 3;
  const int npts[3] = {(int)n, 1, 1};
  int *map = malloc(n * sizeof(int));
  double *host = calloc(2 * n, sizeof(double)), *box = rarray(2 * n), *ref = rarray(2 * n);
  for (size_t i = 0; i < n; i++) {
    map[i] = (int)(n - 1 - i);
    ref[2 * i] = box[2 * map[i]];
    ref[2 * i + 1] = box[2 * map[i] + 1];
  }
  assert(res_natural_prepare(7, map, npts, n));
  assert(pw_gpu_ensure_xc_c(4 * n * sizeof(double)));
  assert(res_sample_buffer(&res[RES_RHO_G], 2 * pw_res_sample_count(n)));
  offloadMemcpyAsyncHtoD(xc_dev, box, 2 * n * sizeof(double), stream);
  res_register_natural_lazy(host, n, xc_dev);
  assert(res_samples_match(&res[RES_RHO_G], ref));
  max_gpu_memory_bytes = mock_limit = current_gpu_allocated_bytes;
  mock_reset_trace();
  const size_t allocations = mock_allocations;
  pw_gpu_res_ensure_host_c(RES_RHO_G, host + 2 * (n - 1));
  same(host, ref, 2 * n, "natural multi-chunk sync");
  assert(mock_allocations == allocations && mock_download_bytes == 2 * n * sizeof(double));
  assert(res_verify_full(&res[RES_RHO_G], host));
  pw_gpu_res_set_lazy_c(0);
  finish();
  max_gpu_memory_bytes = mock_limit = 0;
  free(map); free(host); free(box); free(ref);
}

// Stage 4: XC and potential_pw2rs in the natural order on a full grid, bitwise
// equal to the sorted order, with separable g (axis tables) and with g of a
// cell that couples the axes (full copy). The sorted g array is released.
static void full_grid_g(const int separable) {
  full_grid();
  const int n = nr_[0], n0 = dims[0][0], n1 = dims[0][1], n2 = dims[0][2];
  for (int i = 0; i < n; i++) {
    const int p = map_[0][i], x = p % n0, y = (p / n0) % n1, z = p / (n0 * n1);
    const double gx = 0.31 * (2 * x <= n0 ? x : x - n0);
    const double gy = -0.27 * (2 * y <= n1 ? y : y - n1); // -0.0 at y = 0
    const double gz = 0.23 * (2 * z <= n2 ? z : z - n2);
    g_fine[3 * i] = separable ? gx : gx + 0.05 * gy;
    g_fine[3 * i + 1] = gy;
    g_fine[3 * i + 2] = gz;
  }
}

static void natural_xc(void) {
  for (int sep = 1; sep >= 0; sep--) {
    full_grid_g(sep);
    start(0);
    const bool nat = opt_res_natural_xc;
    const size_t n = (size_t)ng_[0];
    double *r[NLEV], *pot[NLEV], *ref_pot[NLEV];
    const double rscale[NLEV] = {1.0, 8.0, 0.5};
    for (int l = 0; l < NLEV; l++) {
      r[l] = rarray((size_t)nr_[l]);
      for (int i = 0; i < nr_[l]; i++) r[l][i] = 0.5 + 0.2 * r[l][i];
      pot[l] = rarray((size_t)nr_[l]);
      ref_pot[l] = rarray((size_t)nr_[l]);
    }
    double rg[128] = {0}, rr[64] = {0}, ref_g[128], ref_r[64];
    double vxc[64], ref_vxc[64], exc, ref_exc;
    density_reference(r, ref_g, ref_r);
    opt_res_natural_xc = false;
    assert(xc_call(ref_g, ref_r, ref_vxc, &ref_exc));
    pw_gpu_res_ensure_host_c(RES_VXC, ref_vxc);
    res_invalidate(RES_VXC);
    potential_reference(ref_vxc, ref_pot, rscale);
    double grad[3][64], grad_ref[3][64];
    pw_gpu_sfffc_grad3(ref_g, grad_ref[0], grad_ref[1], grad_ref[2], map_[0], g_fine,
                       dims[0], ng_[0], 1, 1);
    opt_res_natural_xc = nat;
    for (int pass = 0; pass < 3; pass++) {
      assert(density_fused(r, rg, rr));
      const unsigned long before = res_stats.nat_xc, dz_before = res_stats.dz_used;
      mock_nat_deriv_n = mock_nat_deriv_acc_n = mock_nat_grad2_n = mock_nat_div2_n = 0;
      // pass 2: the density is not resident; its sorted host copy is scattered.
      if (pass == 2) pw_gpu_res_invalidate_c();
      mock_reset_trace();
      if (pass == 2) {
        assert(xc_call(ref_g, ref_r, vxc, &exc));
      } else {
        assert(xc_call(rg, rr, vxc, &exc));
      }
      pw_gpu_res_ensure_host_c(RES_VXC, vxc);
      xc_same(vxc, ref_vxc, n, "natural XC vxc");
      xc_same(&exc, &ref_exc, 1, "natural XC energy");
      grid_cache_entry *e = NULL;
      for (int i = 0; i < PW_GPU_GRID_CACHE_SIZE; i++)
        if (grid_cache[i].grid_id == 1) e = &grid_cache[i];
      assert(e != NULL);
      if (nat) {
        assert(res_stats.nat_xc == before + 1 && res_stats.nat_gt_mode == (sep ? 1 : 2));
        // Stage 6: pass 1 reads d rho/dz from its density (pass 0 built the g tables).
        const bool dz = opt_res_xc_dz && pass == 1;
        assert(res_stats.dz_used == dz_before + (dz ? 1u : 0u));
        const size_t singles = opt_res_xc_pairs ? 1 : 3;
        assert(mock_nat_deriv_n == (dz ? 0 : singles) * n && mock_nat_deriv_acc_n == singles * n);
        assert(mock_nat_grad2_n == (opt_res_xc_pairs ? n : 0) &&
               mock_nat_div2_n == (opt_res_xc_pairs ? n : 0));
        for (size_t i = 0; i < mock_operation_count; i++)
          assert(mock_operations[i].kind != MOCK_SCATTER_DERIV &&
                 mock_operations[i].kind != MOCK_GATHER_DERIV);
        assert(e->g_dev == NULL && e->gt_mode == (sep ? 1 : 2));
        assert(e->gt_bytes == sizeof(double) *
               (sep ? (size_t)(dims[0][0] + dims[0][1] + dims[0][2]) : 3 * n));
        if (pass > 0) assert(mock_upload_bytes == (pass == 2 ? 3 * n * sizeof(double) : 0));
      } else {
        assert(res_stats.nat_xc == before && e->gt_mode == 0);
      }
      // potential_pw2rs: the coarse levels are read from the auxiliary FFT box.
      const unsigned long pots = res_stats.nat_potential;
      for (int l = 0; l < NLEV; l++) memset(pot[l], 0, sizeof(double) * (size_t)nr_[l]);
      res_invalidate(RES_VXC);
      assert(potential_fused(ref_vxc, pot, rscale));
      for (int l = 0; l < NLEV; l++) same(pot[l], ref_pot[l], (size_t)nr_[l], "natural potential");
      assert(res_stats.nat_potential == pots + (nat && opt_res_natural_g ? 1u : 0u));
    }
    // A sorted consumer (legacy gradient) re-creates the sorted g array on demand.
    pw_gpu_sfffc_grad3(ref_g, grad[0], grad[1], grad[2], map_[0], g_fine, dims[0],
                       ng_[0], 1, 1);
    for (int d = 0; d < 3; d++) same(grad[d], grad_ref[d], n, "sorted gradient after natural XC");
    // The natural XC keeps working next to the re-created sorted copy.
    assert(density_fused(r, rg, rr));
    assert(xc_call(rg, rr, vxc, &exc));
    pw_gpu_res_ensure_host_c(RES_VXC, vxc);
    xc_same(vxc, ref_vxc, n, "natural XC next to sorted g");
    // A rho_g without Hermitian symmetry (and the Nyquist plane of the even x
    // axis): the paired transforms still return exactly the real parts that
    // the sorted path takes of each gradient.
    double skew_g[128], skew_vxc[64], skew_exc;
    for (size_t i = 0; i < 2 * n; i++) skew_g[i] = ref_g[i] * (1.0 + 0.3 * rnd()) + 0.01 * rnd();
    opt_res_natural_xc = false;
    assert(xc_call(skew_g, ref_r, skew_vxc, &skew_exc));
    pw_gpu_res_ensure_host_c(RES_VXC, skew_vxc);
    opt_res_natural_xc = nat;
    assert(xc_call(skew_g, ref_r, vxc, &exc));
    pw_gpu_res_ensure_host_c(RES_VXC, vxc);
    xc_same(vxc, skew_vxc, n, "natural XC, non-Hermitian rho_g");
    xc_same(&exc, &skew_exc, 1, "natural XC energy, non-Hermitian rho_g");
    pw_gpu_res_set_lazy_c(0);
    finish();
    for (int l = 0; l < NLEV; l++) { free(r[l]); free(pot[l]); free(ref_pot[l]); }
  }
}

// Stage 6: d rho/dz from the density's inverse FFT, and every condition under
// which the XC computes its own z gradient instead.
static int xc_call_g(const double *rho_g, const double *rho_r, double *vxc, double *exc,
                     const double *g) {
  return pw_gpu_xc_full_pipeline_configured_c(rho_g, rho_r, vxc, exc, map_[0], g,
      dims[0], ng_[0], nmaps_, 1.0 / nr_[0], 1, 1e-10, .804, .066725,
      .2195149727645171, 1.0, 1.0, 1e-10, 0.0, 0);
}

static void near(const double *a, const double *b, const size_t n, const char *what) {
  double big = 0.0;
  for (size_t i = 0; i < n; i++) big = fmax(big, fabs(b[i]));
  for (size_t i = 0; i < n; i++) {
    if (!(fabs(a[i] - b[i]) <= 1e-13 * (1.0 + big))) {
      fprintf(stderr, "%s: mismatch at %zu: %.17g != %.17g\n", what, i, a[i], b[i]);
      abort();
    }
  }
}

static void natural_dz(void) {
  full_grid_g(1);
  start(0);
  assert(opt_res_natural_xc);
  const bool on = opt_res_xc_dz;
  const size_t n = (size_t)ng_[0];
  double *r[NLEV];
  for (int l = 0; l < NLEV; l++) {
    r[l] = rarray((size_t)nr_[l]);
    for (int i = 0; i < nr_[l]; i++) r[l][i] = 0.5 + 0.2 * r[l][i];
  }
  double rg[128] = {0}, rr[64] = {0}, ref_g[128], ref_r[64];
  double vxc[64], ref_vxc[64], exc, ref_exc, grad[3][64], dz[64];
  density_reference(r, ref_g, ref_r);
  opt_res_natural_xc = false;
  assert(xc_call(ref_g, ref_r, ref_vxc, &ref_exc));
  pw_gpu_res_ensure_host_c(RES_VXC, ref_vxc);
  res_invalidate(RES_VXC);
  opt_res_natural_xc = true;
  pw_gpu_sfffc_grad3(ref_g, grad[0], grad[1], grad[2], map_[0], g_fine, dims[0], ng_[0], 1, 1);
#define XC_OK(rho_g, rho_r, what) do { \
    assert(xc_call(rho_g, rho_r, vxc, &exc)); \
    pw_gpu_res_ensure_host_c(RES_VXC, vxc); \
    xc_same(vxc, ref_vxc, n, what); \
    xc_same(&exc, &ref_exc, 1, what); \
    res_invalidate(RES_VXC); \
  } while (0)
  // No g tables before the first natural XC: plain density, bitwise rho_r.
  assert(density_fused(r, rg, rr));
  assert(res_stats.dz_made == 0 && mock_rho_dz_n == 0);
  pw_gpu_res_ensure_host_c(RES_RHO_R, rr);
  same(rr, ref_r, n, "density without g tables");
  XC_OK(rg, rr, "first XC");
  assert(res_stats.dz_used == 0);
  // Steady state: one transform less in the XC, no allocation, no upload.
  for (int step = 0; step < 2; step++) {
    const size_t allocations = mock_allocations;
    const int ffts = mock_fft_calls;
    assert(density_fused(r, rg, rr));
    const int density_ffts = mock_fft_calls - ffts;
    assert(res_stats.dz_made == (on ? step + 1u : 0u) && mock_rho_dz_n == (on ? (step + 1) * n : 0));
    if (on) {
      offloadMemcpyAsyncDtoH(dz, xc_dev_hi + n, n * sizeof(double), stream);
      offloadStreamSynchronize(stream);
      near(dz, grad[2], n, "d rho/dz of the density");
    }
    mock_nat_deriv_n = 0;
    const int xc_ffts = mock_fft_calls;
    const size_t up = mock_upload_bytes;
    XC_OK(rg, rr, "XC with d rho/dz");
    assert(mock_upload_bytes == up);
    assert(res_stats.dz_used == (on ? step + 1u : 0u));
    const bool skip = on && !opt_res_verify;
    assert(mock_nat_deriv_n == (skip ? 0 : opt_res_xc_pairs ? 1 : 3) * n);
    assert(mock_fft_calls - xc_ffts == (skip ? 4 : opt_res_xc_pairs ? 5 : 7));
    assert(mock_allocations == allocations && density_ffts > 0);
  }
  pw_gpu_res_ensure_host_c(RES_RHO_R, rr);
  if (on) near(rr, ref_r, n, "rho_r with d rho/dz");
  else same(rr, ref_r, n, "rho_r without d rho/dz");
  unsigned long used = res_stats.dz_used;
  // RHO_R or RHO_G not resident: the XC reads host inputs, its own z gradient.
  for (int id = RES_RHO_G; id <= RES_RHO_R; id++) {
    assert(density_fused(r, rg, rr));
    pw_gpu_res_ensure_host_c(id, id == RES_RHO_G ? (void *)rg : (void *)rr);
    res_invalidate(id);
    XC_OK(rg, rr, "XC with one input uploaded");
    assert(res_stats.dz_used == used);
  }
  // Rebuilt g tables (other g array): new generation, d rho/dz is not used.
  double *g_copy = malloc(3 * n * sizeof(double));
  memcpy(g_copy, g_fine, 3 * n * sizeof(double));
  assert(density_fused(r, rg, rr));
  assert(xc_call_g(rg, rr, vxc, &exc, g_copy));
  pw_gpu_res_ensure_host_c(RES_VXC, vxc);
  xc_same(vxc, ref_vxc, n, "XC with rebuilt g tables");
  res_invalidate(RES_VXC);
  assert(res_stats.dz_used == used);
  // The rebuilt tables serve the next density.
  assert(density_fused(r, rg, rr));
  assert(xc_call_g(rg, rr, vxc, &exc, g_copy));
  pw_gpu_res_ensure_host_c(RES_VXC, vxc);
  xc_same(vxc, ref_vxc, n, "XC with the new tables");
  res_invalidate(RES_VXC);
  assert(res_stats.dz_used == used + (on ? 1u : 0u));
  used = res_stats.dz_used;
  free(g_copy);
  // Consumed once: a second XC of the same (host-valid) arrays recomputes.
  if (!opt_res_lazy_density) {
    assert(density_fused(r, rg, rr));
    XC_OK(rg, rr, "first XC of a density");     // rebuilds the tables for g_fine
    assert(res_stats.dz_used == used);
    assert(density_fused(r, rg, rr));
    XC_OK(rg, rr, "XC with d rho/dz again");
    assert(res_stats.dz_used == used + (on ? 1u : 0u));
    used = res_stats.dz_used;
    XC_OK(rg, rr, "second XC of the same density");
    assert(res_stats.dz_used == used);
  }
  // Grid invalidation between density and XC.
  XC_OK(ref_g, ref_r, "tables for g_fine");
  assert(density_fused(r, rg, rr));
  pw_gpu_grid_invalidate(1);
  XC_OK(rg, rr, "XC after grid invalidation");
  assert(res_stats.dz_used == used);
  // CP2K_PW_GPU_RES_VERIFY: a corrupted d rho/dz is detected and replaced.
  if (opt_res_verify && on) {
    assert(density_fused(r, rg, rr));
    XC_OK(rg, rr, "tables after invalidation");
    assert(density_fused(r, rg, rr));
    assert(res_dz.epoch == xc_epoch);
    double bad = 1.0;
    offloadMemcpyAsyncHtoD(xc_dev_hi + n + 3, &bad, sizeof(double), stream);
    const unsigned long bads = res_stats.dz_bad;
    XC_OK(rg, rr, "XC with a corrupted d rho/dz");
    assert(res_stats.dz_bad == bads + 1 && res_stats.dz_used == used + 1);
  }
  assert(res_stats.dz_bad == 0 || opt_res_verify);
  if (!on) assert(res_stats.dz_made == 0 && res_stats.dz_used == 0 && mock_rho_dz_n == 0);
#undef XC_OK
  pw_gpu_res_set_lazy_c(0);
  finish();
  for (int l = 0; l < NLEV; l++) free(r[l]);
}

// One SCF step after warm-up uploads only the real-space density inputs and
// the small fingerprint/DDAPC data: the target map, its g-vectors and the
// coarse indices (sorted and natural) stay cached.
static void natural_traffic(void) {
  full_grid();
  start(0);
  const int n = ng_[0], nr = nr_[0];
  double *r[NLEV], *pot[NLEV], *ref_pot[NLEV];
  const double rscale[NLEV] = {1.0, 8.0, 0.5};
  for (int l = 0; l < NLEV; l++) {
    r[l] = rarray((size_t)nr_[l]);
    for (int i = 0; i < nr_[l]; i++) r[l][i] = 0.5 + 0.2 * r[l][i];
    pot[l] = rarray((size_t)nr_[l]);
    ref_pot[l] = rarray((size_t)nr_[l]);
  }
  double ref_g[128], ref_r[64], ref_vxc[64], ref_exc, ref_tot[128], ref_vh[64], ref_e;
  density_reference(r, ref_g, ref_r);
  hartree_in h = hartree_inputs();
  memcpy(h.rho_g, ref_g, sizeof(double) * 2 * (size_t)n);
  hartree_reference(&h, ref_tot, ref_vh, &ref_e);
  assert(xc_call(ref_g, ref_r, ref_vxc, &ref_exc));
  pw_gpu_res_ensure_host_c(RES_VXC, ref_vxc);
  double ref_vtot[64];
  for (int i = 0; i < nr; i++) ref_vtot[i] = ref_vxc[i] * h.dvol + ref_vh[i];
  potential_reference(ref_vtot, ref_pot, rscale);
  double *rg = rarray(2 * (size_t)n), *rr = rarray((size_t)nr);
  free(h.rho_g);
  h.rho_g = rg;
  double tot[128], vh[64], vxc[64], exc, e;
  const bool lazy = opt_res_lazy_density;
  for (int step = 0; step < 3; step++) {
    size_t up = mock_upload_bytes;
    assert(density_fused(r, rg, rr));
    const size_t up_density = mock_upload_bytes - up;
    up = mock_upload_bytes;
    assert(hartree_fused(&h, tot, vh, &e, 0));
    up = mock_upload_bytes;
    assert(xc_call(rg, rr, vxc, &exc));
    const size_t up_xc = mock_upload_bytes - up;
    assert(pw_gpu_res_vtot_prepare_c(vh, vxc, (size_t)nr, h.dvol));
    if (lazy) {
      assert(pw_gpu_res_vtot_skip_host_c());
      assert(pw_gpu_res_vtot_skip_host_c());
    } else {
      for (int i = 0; i < nr; i++) vxc[i] = vxc[i] * h.dvol;
      for (int i = 0; i < nr; i++) vxc[i] = vxc[i] + vh[i];
      assert(pw_gpu_res_vtot_commit_c(vxc, (size_t)nr));
    }
    up = mock_upload_bytes;
    assert(potential_fused(vxc, pot, rscale));
    const size_t up_potential = mock_upload_bytes - up;
    pw_gpu_res_ensure_host_c(RES_V_H_R, vh);
    pw_gpu_res_ensure_host_c(RES_VXC, vxc);
    same(vh, ref_vh, (size_t)nr, "traffic v_h");
    xc_same(&exc, &ref_exc, 1, "traffic exc");
    for (int l = 0; l < NLEV; l++) xc_same(pot[l], ref_pot[l], (size_t)nr_[l], "traffic potential");
    if (step > 0) {
      size_t inputs = 0;
      for (int l = 0; l < NLEV; l++) inputs += sizeof(double) * (size_t)nr_[l];
      // Natural order: fingerprint coordinates (lazy) or the chunked download map.
      const size_t coords = sizeof(int) * (lazy ? pw_res_sample_count((size_t)n) : (size_t)n);
      assert(up_xc == 0 && up_potential == 0);
      if (up_density != inputs + (opt_res_natural_g ? coords : 0))
        fprintf(stderr, "density uploads %zu, expected %zu\n", up_density,
                inputs + (opt_res_natural_g ? coords : 0));
      assert(up_density == inputs + (opt_res_natural_g ? coords : 0));
    }
  }
  pw_gpu_res_set_lazy_c(0);
  finish();
  for (int l = 0; l < NLEV; l++) { free(r[l]); free(pot[l]); free(ref_pot[l]); }
  free(rr);
  hartree_free(&h);
}

int main(int argc, char **argv) {
  assert(argc == 2);
  if (!strcmp(argv[1], "chain")) natural_chain();
  else if (!strcmp(argv[1], "lifetime")) natural_lifetime();
  else if (!strcmp(argv[1], "memory")) natural_memory();
  else if (!strcmp(argv[1], "chunked_sync")) natural_chunked_sync();
  else if (!strcmp(argv[1], "traffic")) natural_traffic();
  else if (!strcmp(argv[1], "xc")) natural_xc();
  else if (!strcmp(argv[1], "dz")) natural_dz();
  else assert(!"unknown natural test");
  for (int l = 0; l < NLEV; l++) { free(map_[l]); free(gidx_[l]); }
  free(g_fine); free(gsq_fine);
  puts("PASS");
  return 0;
}
'''


def main():
    hermes.source_checks()
    cc = os.environ.get("CC", "gcc").split()
    cxx = os.environ.get("CXX", "g++").split()
    blas = os.environ.get("BLAS_LIBS", "-lblas").split()
    flags = ["-std=c11", "-Wall", "-Wextra", "-Werror", "-Wpedantic", "-fopenmp", "-ffp-contract=off",
             "-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    with tempfile.TemporaryDirectory(prefix=".build-", dir=HERE) as tmp:
        build = Path(tmp)
        gpu = build / "src/pw/gpu"
        gpu.mkdir(parents=True)
        for path in (ROOT / "src/pw/gpu").iterdir():
            if path.suffix in (".h", ".c"):
                shutil.copy2(path, gpu)
        off = build / "src/offload"
        off.mkdir()
        for name in ("offload_runtime.h", "offload_fft.h", "offload_library.h"):
            (off / name).write_text('#include "mock_offload.h"\n')
        (off / "offload_library.h").write_text('#include "mock_offload.h"\n' + NATURAL_MOCK)
        inc = ["-I", str(HERMES), "-I", str(gpu), "-I", str(HERE)]
        natural_test = build / "test_natural.c"
        natural_test.write_text(NATURAL_CASES)
        exes = {}
        for test, src in (("test_pw_gpu", HERMES / "test_pw_gpu.c"), ("test_res", HERE / "test_res.c"),
                          ("test_natural", natural_test)):
            exes[test] = build / test
            subprocess.run(cc + flags + inc + ["-D__OFFLOAD", str(src), str(HERMES / "mock_offload.c"),
                           *blas, "-lm", "-o", str(exes[test])], check=True)
        for defines in ([], ["-D__OFFLOAD", "-D__NO_OFFLOAD_PW"]):
            subprocess.run(cc + flags + inc + defines + ["-c", str(gpu / "pw_gpu_internal.c"),
                           "-o", str(build / "x.o")], check=True)
        env = {k: v for k, v in os.environ.items() if k not in hermes.RANK_NAMES and
               not k.startswith(("CP2K_GPU_", "CP2K_PW_GPU_"))}
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                   UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
        source = (ROOT / "src/pw/gpu/pw_gpu_kernels.cu").read_text()
        source, count = re.subn(r"(\w+)<<<(.*?)>>>\(", r"emulate(\1, \2, ", source, flags=re.S)
        assert count == 34, count  # + 5 natural XC kernels (stage 4) + 5 paired/fused (stage 5) + 1 (stage 6)
        prelude = (HERMES / "kernel_body_tests.py").read_text().split("prelude = '''", 1)[1].split("'''", 1)[0]
        body = (HERMES / "test_kernel_bodies.cpp").read_text()
        body = body.replace("int main() {", "static void hermes_main() {", 1)
        body += NATURAL_KERNEL_TESTS
        body += (HERE / "test_res_kernels.cpp").read_text().replace(
            "hermes_main();", "hermes_main(); natural_kernel_tests();")
        path = gpu / "emulated_kernels.cpp"
        path.write_text(prelude + source + body)
        cxxflags = [f for f in flags if f != "-std=c11"] + ["-std=c++17"]
        subprocess.run(cxx + cxxflags + inc + ["-D__OFFLOAD", str(path), "-o", str(build / "kern")], check=True)
        subprocess.run([str(build / "kern")], env=env, check=True, timeout=60)
        print("PASS emulated kernel bodies (Hermes + residency)")
        n = 0
        for name, args, overrides, status, notices in hermes.cases():
            r = subprocess.run([str(exes["test_pw_gpu"]), *args], env=env | overrides, text=True,
                               capture_output=True, timeout=60)
            if status == -1:
                assert r.returncode != 0 and "host fallback is disabled" in r.stderr, (name, r)
            else:
                assert r.returncode == status, (name, r.stdout, r.stderr)
                assert r.stdout.strip() == "PASS", (name, r)
            assert r.stderr.count("falling back to host.") == notices, (name, r.stderr)
            assert "AddressSanitizer" not in r.stderr and "runtime error:" not in r.stderr, r.stderr
            n += 1
        print(f"PASS {n} Hermes C mock cases")
        cases = [(name, {"CP2K_PW_GPU_RES_LAZY_DENSITY": "0"}) for name in RES_CASES]
        # Hermes's lazy cases cover the lazy density alone; the stage 3a cases the window,
        # the density token and the lazy Vxc/V_H.
        cases += [(name, {"CP2K_PW_GPU_RES_LAZY_POT": "0"})
                  for name in ("lazy_density", "integrate", "chain", "lazy_vtot", "borrowed")]
        cases += [(name, {}) for name in ("lazy_grant", "lazy_pot")]
        for name, overrides in cases:
            for threads in ("1", "3"):
                r = subprocess.run([str(exes["test_res"]), name],
                                   env=env | overrides | {"OMP_NUM_THREADS": threads},
                                   text=True, capture_output=True, timeout=120)
                assert r.returncode == 0 and r.stdout.strip() == "PASS", (name, r.stdout, r.stderr)
                assert "AddressSanitizer" not in r.stderr and "runtime error:" not in r.stderr, r.stderr
            mode = "eager" if "CP2K_PW_GPU_RES_LAZY_DENSITY" in overrides else \
                "lazy density" if overrides else "lazy window"
            print(f"PASS residency {name} ({mode}, OMP=1/3)")
        natural_cases = [
            ("chain", {}),
            ("chain", {"CP2K_PW_GPU_RES_LAZY_DENSITY": "0", "CP2K_PW_GPU_RES_LAZY_POT": "0"}),
            ("chain", {"CP2K_PW_GPU_RES_NATURAL_G": "0"}),
            ("chain", {"CP2K_PW_GPU_RES_VERIFY": "1"}),
            ("lifetime", {}),
            ("memory", {}),
            ("chunked_sync", {}),
            ("traffic", {}),
            ("traffic", {"CP2K_PW_GPU_RES_LAZY_DENSITY": "0", "CP2K_PW_GPU_RES_LAZY_POT": "0"}),
            ("traffic", {"CP2K_PW_GPU_RES_NATURAL_G": "0"}),
            ("xc", {}),
            ("xc", {"CP2K_PW_GPU_RES_LAZY_DENSITY": "0", "CP2K_PW_GPU_RES_LAZY_POT": "0"}),
            ("xc", {"CP2K_PW_GPU_RES_NATURAL_XC": "0"}),
            ("xc", {"CP2K_PW_GPU_RES_NATURAL_G": "0"}),
            ("xc", {"CP2K_PW_GPU_RES_XC_PAIRS": "0"}),
            ("traffic", {"CP2K_PW_GPU_RES_XC_PAIRS": "0"}),
            ("chain", {"CP2K_PW_GPU_RES_XC_PAIRS": "0"}),
            ("dz", {}),
            ("dz", {"CP2K_PW_GPU_RES_LAZY_DENSITY": "0", "CP2K_PW_GPU_RES_LAZY_POT": "0"}),
            ("dz", {"CP2K_PW_GPU_RES_VERIFY": "1"}),
            ("dz", {"CP2K_PW_GPU_RES_XC_DZ": "0"}),
            ("dz", {"CP2K_PW_GPU_RES_XC_PAIRS": "0"}),
            ("xc", {"CP2K_PW_GPU_RES_XC_DZ": "0"}),
            ("traffic", {"CP2K_PW_GPU_RES_XC_DZ": "0"}),
        ]
        for name, overrides in natural_cases:
            for threads in ("1", "3"):
                r = subprocess.run([str(exes["test_natural"]), name],
                                   env=env | overrides | {"OMP_NUM_THREADS": threads},
                                   text=True, capture_output=True, timeout=120)
                assert r.returncode == 0 and r.stdout.strip() == "PASS", (name, overrides, r.stdout, r.stderr)
                assert "AddressSanitizer" not in r.stderr and "runtime error:" not in r.stderr, r.stderr
            print(f"PASS natural FFT {name} ({overrides or 'defaults'}, OMP=1/3)")
        run_fortran_tests(build, cc, flags, inc, blas, env)
        run_grid_tests(build, cc, flags, inc, blas, env)


if __name__ == "__main__":
    main()
