/* SPDX-License-Identifier: GPL-2.0-or-later */
// Device residency against the original call sequences and lazy host access.
//
// The reference for every fused path is the unfused sequence the Fortran code
// executes: pw_gpu_cfffg_cached / pw_gpu_sfffc_cached (through the same mock
// FFT) plus plain loops for pw_axpy, pw_copy, pw_multiply_with and pw_scale,
// compiled with -ffp-contract=off like the device element functions. Arrays
// must agree bit for bit; only the Hartree energy (a reduction) is compared
// with a tolerance.
#define _POSIX_C_SOURCE 200809L
#include <stddef.h>
static size_t mock_free_memory_value = (size_t)1 << 40;
static size_t mock_free_memory(void) { return mock_free_memory_value; }
#define PW_GPU_TEST_FREE_MEMORY mock_free_memory
#include "pw_gpu_internal.c"
#include "mock_offload.h"
#define PW_GPU_ELEMENTAL static inline
#include "pw_gpu_grid_ops.h"
#undef PW_GPU_ELEMENTAL

extern int mock_res_calls;

// ---------------------------------------------------------------- utilities
static void accounting(void) {
  size_t allocated, peak;
  pw_gpu_get_allocated_bytes(&allocated, &peak);
  assert(allocated == mock_allocated);
  assert(peak == mock_peak);
  assert(max_gpu_memory_bytes == 0 || allocated <= max_gpu_memory_bytes);
}

static void start(const int mb) {
  assert(mock_allocated == 0 && mock_live_plans == 0);
  mock_peak = mock_allocations = 0;
  mock_uploads = mock_fft_calls = 0;
  pw_gpu_set_memory_limit(mb);
  mock_limit = max_gpu_memory_bytes;
  pw_gpu_init();
  mock_reset_trace();
  accounting();
}

static void finish(void) {
  accounting();
  pw_gpu_finalize();
  assert(mock_allocated == 0 && mock_live_plans == 0);
  assert(mock_live_streams == 0 && mock_live_events == 0);
  assert(res_small_dev == NULL);
  for (int i = 0; i < RES_NSLOTS; i++) assert(res[i].owned_bytes == 0);
}

static void same(const double *a, const double *b, const size_t n, const char *what) {
  for (size_t i = 0; i < n; i++) {
    if (memcmp(&a[i], &b[i], sizeof(double)) != 0) {
      fprintf(stderr, "%s: bitwise mismatch at %zu: %.17g != %.17g\n", what, i, a[i], b[i]);
      abort();
    }
  }
}

static uint64_t rng = 88172645463325252ULL;
static double rnd(void) {
  rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
  return (double)(rng >> 11) / 9007199254740992.0 - 0.5;
}

// ------------------------------------------------------------- test grids
// Level 0: the fine (target, XC) grid. Level 1: a coarse grid whose G vectors
// are a subset of the fine ones (gidx, one-based). Level 2: a smaller one.
#define NLEV 3
static const int dims[NLEV][3] = {{4, 3, 5}, {3, 2, 3}, {2, 2, 2}};
static int nr_[NLEV], ng_[NLEV];
static int nmaps_ = 1;
static int *map_[NLEV], *gidx_[NLEV];
static double *g_fine, *gsq_fine;

static void permutation(int *p, const int n) {
  for (int i = 0; i < n; i++) p[i] = i;
  for (int i = n - 1; i > 0; i--) {
    const int j = (int)((rnd() + 0.5) * (i + 1)) % (i + 1);
    const int t = p[i]; p[i] = p[j]; p[j] = t;
  }
}

static void grids(const int nmaps) {
  nmaps_ = nmaps;
  const int ngs[NLEV] = {24, 7, 4};
  for (int l = 0; l < NLEV; l++) {
    nr_[l] = dims[l][0] * dims[l][1] * dims[l][2];
    ng_[l] = ngs[l];
    free(map_[l]);
    map_[l] = malloc(sizeof(int) * (size_t)nmaps * (size_t)ng_[l]);
    int perm[64];
    permutation(perm, nr_[l]);
    // Disjoint positions for both maps.
    for (int m = 0; m < nmaps; m++)
      for (int i = 0; i < ng_[l]; i++) map_[l][m * ng_[l] + i] = perm[m * ng_[l] + i];
    free(gidx_[l]);
    gidx_[l] = NULL;
    if (l > 0) {
      gidx_[l] = malloc(sizeof(int) * (size_t)ng_[l]);
      permutation(perm, ng_[0]);
      for (int i = 0; i < ng_[l]; i++) gidx_[l][i] = perm[i] + 1;
    }
  }
  free(g_fine); free(gsq_fine);
  g_fine = malloc(sizeof(double) * 3 * (size_t)ng_[0]);
  gsq_fine = malloc(sizeof(double) * (size_t)ng_[0]);
  // Sorted |G|^2 with G=0 first, as in CP2K.
  for (int i = 0; i < ng_[0]; i++) {
    const double len = 0.15 * i;
    g_fine[3 * i] = len * 0.6; g_fine[3 * i + 1] = len * 0.48; g_fine[3 * i + 2] = len * 0.64;
    gsq_fine[i] = g_fine[3 * i] * g_fine[3 * i] + g_fine[3 * i + 1] * g_fine[3 * i + 1] +
                  g_fine[3 * i + 2] * g_fine[3 * i + 2];
  }
}

static double *rarray(const size_t n) {
  double *a = malloc(sizeof(double) * n);
  for (size_t i = 0; i < n; i++) a[i] = rnd();
  return a;
}

// --------------------------------------------------- original sequences
// density_rs2pw, pw_interp: pw_zero, {pw_transfer, pw_axpy}*, pw_transfer
static void density_reference(double *const *r, double *rho_g, double *rho_r) {
  memset(rho_g, 0, sizeof(double) * 2 * (size_t)ng_[0]);
  double tmp[2 * 64];
  for (int l = 0; l < NLEV; l++) {
    pw_gpu_cfffg_cached(r[l], tmp, map_[l], dims[l], ng_[l], nmaps_, 1.0 / nr_[l], l + 1);
    for (int i = 0; i < ng_[l]; i++) {
      const int j = l == 0 ? i : gidx_[l][i] - 1;
      rho_g[2 * j] = rho_g[2 * j] + tmp[2 * i];
      rho_g[2 * j + 1] = rho_g[2 * j + 1] + tmp[2 * i + 1];
    }
  }
  pw_gpu_sfffc_cached(rho_g, rho_r, map_[0], dims[0], ng_[0], nmaps_, 1.0, 1);
}

static int density_fused(double *const *r, double *rho_g, double *rho_r) {
  void *rp[NLEV], *mp[NLEV], *gp[NLEV];
  int npts[3 * NLEV], ids[NLEV], nm[NLEV];
  double scale[NLEV];
  for (int l = 0; l < NLEV; l++) {
    rp[l] = r[l]; mp[l] = map_[l]; gp[l] = gidx_[l];
    memcpy(npts + 3 * l, dims[l], sizeof(dims[l]));
    ids[l] = l + 1; nm[l] = nmaps_; scale[l] = 1.0 / nr_[l];
  }
  return pw_gpu_res_density_c(NLEV, rp, npts, ng_, nm, ids, mp, gp, scale, dims[0],
                              ng_[0], nmaps_, 1, map_[0], rho_g, rho_r);
}

// potential_pw2rs, pw_interp, auxbas = level 0
static void potential_reference(const double *v, double *const *r, const double *rscale) {
  double gaux[2 * 64], gl[2 * 64];
  pw_gpu_cfffg_cached(v, gaux, map_[0], dims[0], ng_[0], nmaps_, 1.0 / nr_[0], 1);
  memcpy(r[0], v, sizeof(double) * (size_t)nr_[0]);
  for (int l = 1; l < NLEV; l++) {
    for (int i = 0; i < ng_[l]; i++) {
      gl[2 * i] = gaux[2 * (gidx_[l][i] - 1)];
      gl[2 * i + 1] = gaux[2 * (gidx_[l][i] - 1) + 1];
    }
    pw_gpu_sfffc_cached(gl, r[l], map_[l], dims[l], ng_[l], nmaps_, 1.0, l + 1);
    for (int i = 0; i < nr_[l]; i++) r[l][i] = rscale[l] * r[l][i];
  }
}

static int potential_fused(const double *v, double *const *r, const double *rscale) {
  void *rp[NLEV], *mp[NLEV], *gp[NLEV];
  int npts[3 * NLEV], ids[NLEV], nm[NLEV];
  for (int l = 0; l < NLEV; l++) {
    rp[l] = r[l]; mp[l] = map_[l]; gp[l] = gidx_[l];
    memcpy(npts + 3 * l, dims[l], sizeof(dims[l]));
    ids[l] = l + 1; nm[l] = nmaps_;
  }
  return pw_gpu_res_potential_c(v, NLEV, rp, npts, ng_, nm, ids, mp, gp, rscale, 0,
                                1.0 / nr_[0]);
}

// Hartree + DDAPC decoupling (qs_ks_build_kohn_sham_matrix with MULTIPOLE).
typedef struct {
  double *core, *rho_g, *infl;
  double r[3 * 2], radii[2], cv[2 * 2];
  double g0_delta, gcut2, vol, dvol;
  int natoms, nradii;
} hartree_in;

static void hartree_reference(const hartree_in *h, double *rho_tot, double *v_r,
                              double *energy) {
  const int n = ng_[0];
  double v[2 * 64] = {0};
  for (int i = 0; i < 2 * n; i++) rho_tot[i] = h->core[i] + h->rho_g[i];
  for (int i = 0; i < n; i++) { // pw_multiply_with (copy of rho_tot)
    const double a = rho_tot[2 * i], b = rho_tot[2 * i + 1];
    const double c = h->infl[2 * i], d = h->infl[2 * i + 1];
    v[2 * i] = a * c - b * d;
    v[2 * i + 1] = a * d + b * c;
  }
  v[0] = v[0] + h->g0_delta; // complex + real
  v[1] = v[1] + 0.0;
  for (int i = 1; i < n && !(gsq_fine[i] > h->gcut2); i++)
    pw_hartree_elem(v + 2, g_fine + 3, gsq_fine + 1, h->r, h->radii, h->cv, (size_t)(i - 1),
                    (size_t)h->natoms, (size_t)h->nradii, h->gcut2, h->vol);
  double sum = 0.0;
  for (int i = 0; i < 2 * n; i++) sum += rho_tot[i] * v[i];
  *energy = 0.5 * (sum * h->vol);
  pw_gpu_sfffc_cached(v, v_r, map_[0], dims[0], n, nmaps_, 1.0, 1);
  for (int i = 0; i < nr_[0]; i++) v_r[i] = h->dvol * v_r[i];
}

static int hartree_fused(const hartree_in *h, double *rho_tot, double *v_r,
                         double *energy, const int interrupt) {
  const int n = ng_[0];
  int nprefix = 0;
  for (int i = 1; i < n && !(gsq_fine[i] > h->gcut2); i++) nprefix = i;
  nprefix += 1; // G = 0
  if (!pw_gpu_res_hartree_begin_c(h->core, h->rho_g, h->infl, rho_tot, (size_t)n,
                                  (size_t)nprefix, dims[0]))
    return 0;
  if (interrupt) { // another PW GPU operation between the steps
    double tmp[2 * 64], real[64] = {0};
    pw_gpu_cfffg_cached(real, tmp, map_[0], dims[0], ng_[0], nmaps_, 1.0, 1);
  }
  if (!pw_gpu_res_hartree_modify_c(h->g0_delta, 1, g_fine + 3, gsq_fine + 1, h->r,
                                   h->radii, h->cv, 1, (size_t)(n - 1),
                                   (size_t)h->natoms, (size_t)h->nradii, h->gcut2, h->vol))
    return 0;
  double dot;
  if (!pw_gpu_res_hartree_energy_c(&dot)) return 0;
  *energy = 0.5 * (dot * h->vol);
  return pw_gpu_res_hartree_finish_c(v_r, dims[0], n, nmaps_, 1, map_[0], h->dvol);
}

static hartree_in hartree_inputs(void) {
  hartree_in h;
  const int n = ng_[0];
  h.core = rarray(2 * (size_t)n);
  h.rho_g = rarray(2 * (size_t)n);
  h.infl = calloc(2 * (size_t)n, sizeof(double));
  for (int i = 1; i < n; i++) h.infl[2 * i] = 4.0 * 3.14159 / gsq_fine[i];
  for (int i = 0; i < 6; i++) h.r[i] = 3.0 * rnd();
  h.radii[0] = 0.5; h.radii[1] = 1.1;
  for (int i = 0; i < 4; i++) h.cv[i] = rnd();
  h.natoms = 2; h.nradii = 2;
  h.g0_delta = -0.37; h.gcut2 = gsq_fine[9] + 1e-9; h.vol = 27.0; h.dvol = 27.0 / nr_[0];
  return h;
}

static void hartree_free(hartree_in *h) { free(h->core); free(h->rho_g); free(h->infl); }

// ------------------------------------------------------------------ cases
static void density(void) {
  for (int nmaps = 1; nmaps <= 2; nmaps++) {
    grids(nmaps);
    start(0);
    double *r[NLEV];
    for (int l = 0; l < NLEV; l++) r[l] = rarray((size_t)nr_[l]);
    double *ref_g = rarray(2 * (size_t)ng_[0]), *ref_r = rarray((size_t)nr_[0]);
    double *g = rarray(2 * (size_t)ng_[0]), *rr = rarray((size_t)nr_[0]);
    density_reference(r, ref_g, ref_r);
    mock_reset_trace();
    assert(density_fused(r, g, rr));
    same(ref_g, g, 2 * (size_t)ng_[0], "density rho_g");
    same(ref_r, rr, (size_t)nr_[0], "density rho_r");
    // Only the level inputs are uploaded (maps/gidx are cached by now) and
    // only the two results are downloaded.
    size_t up = 0, gidx_bytes = 0;
    for (int l = 0; l < NLEV; l++) up += sizeof(double) * (size_t)nr_[l];
    for (int l = 1; l < NLEV; l++) gidx_bytes += sizeof(int) * (size_t)ng_[l];
    assert(mock_upload_bytes == up + gidx_bytes); // gidx once
    assert(mock_download_bytes == sizeof(double) * (2 * (size_t)ng_[0] + (size_t)nr_[0]));
    assert(res[RES_RHO_G].state == RES_VALID && res[RES_RHO_R].state == RES_VALID);
    mock_assert_idle();
    // Repeated call is identical (cached maps and gidx).
    mock_reset_trace();
    assert(density_fused(r, g, rr));
    assert(mock_upload_bytes == up);
    same(ref_g, g, 2 * (size_t)ng_[0], "density rho_g (2)");
    finish();
    for (int l = 0; l < NLEV; l++) free(r[l]);
    free(ref_g); free(ref_r); free(g); free(rr);
  }
}

static void density_rejects(void) {
  grids(1);
  start(0);
  double *r[NLEV];
  for (int l = 0; l < NLEV; l++) r[l] = rarray((size_t)nr_[l]);
  double *g = rarray(2 * (size_t)ng_[0]), *rr = rarray((size_t)nr_[0]);
  double *g0 = malloc(sizeof(double) * 2 * (size_t)ng_[0]);
  memcpy(g0, g, sizeof(double) * 2 * (size_t)ng_[0]);
  // gidx out of range: detected before any output is changed.
  gidx_[1][3] = ng_[0] + 1;
  assert(!density_fused(r, g, rr));
  same(g0, g, 2 * (size_t)ng_[0], "untouched");
  gidx_[1][3] = 1;
  // Disabled residency returns 0 without device work.
  opt_res_density = false;
  mock_reset_trace();
  assert(!density_fused(r, g, rr) && mock_operation_count == 0);
  opt_res_density = true;
  finish();
  for (int l = 0; l < NLEV; l++) free(r[l]);
  free(g); free(rr); free(g0);
}

static void potential(void) {
  for (int nmaps = 1; nmaps <= 2; nmaps++) {
    grids(nmaps);
    start(0);
    const double rscale[NLEV] = {1.0, 8.0, 1.0};
    double *v = rarray((size_t)nr_[0]);
    double *ref[NLEV], *out[NLEV];
    for (int l = 0; l < NLEV; l++) { ref[l] = rarray((size_t)nr_[l]); out[l] = rarray((size_t)nr_[l]); }
    potential_reference(v, ref, rscale);
    mock_reset_trace();
    assert(potential_fused(v, out, rscale));
    for (int l = 0; l < NLEV; l++) same(ref[l], out[l], (size_t)nr_[l], "potential");
    // Non-resident: the potential is uploaded once; only coarse levels come back.
    assert(mock_upload_bytes == sizeof(double) * (size_t)nr_[0] +
                                sizeof(int) * (size_t)(ng_[1] + ng_[2]));
    assert(mock_download_bytes == sizeof(double) * (size_t)(nr_[1] + nr_[2]));
    finish();
    free(v);
    for (int l = 0; l < NLEV; l++) { free(ref[l]); free(out[l]); }
  }
}

static void hartree(void) {
  for (int nmaps = 1; nmaps <= 2; nmaps++) {
    grids(nmaps);
    start(0);
    hartree_in h = hartree_inputs();
    const int n = ng_[0];
    double ref_tot[2 * 64], tot[2 * 64], ref_v[64], v[64], e_ref, e;
    hartree_reference(&h, ref_tot, ref_v, &e_ref);
    for (int pass = 0; pass < 2; pass++) { // second pass: core and influence resident
      memset(tot, 0, sizeof(tot));
      mock_reset_trace();
      assert(hartree_fused(&h, tot, v, &e, 0));
      same(ref_v, v, (size_t)nr_[0], "hartree v_r");
      int nprefix = 1;
      for (int i = 1; i < n && !(gsq_fine[i] > h.gcut2); i++) nprefix = i + 1;
      same(ref_tot, tot, 2 * (size_t)nprefix, "hartree rho_tot prefix");
      assert(fabs(e - e_ref) <= 1e-14 * fabs(e_ref));
      if (pass == 1) {
        // Only rho_g (not resident without density_rs2pw) and the DDAPC
        // parameters are uploaded.
        size_t active = 0;
        for (int i = 1; i < n && !(gsq_fine[i] > h.gcut2); i++) active++;
        const size_t small = 2 + 4 * active + 3 * (size_t)h.natoms + (size_t)h.nradii +
                             (size_t)(h.natoms * h.nradii);
        assert(mock_upload_bytes == sizeof(double) * (2 * (size_t)n + small));
        assert(res[RES_RHO_CORE].state == RES_VALID && res[RES_INFL].state == RES_VALID);
      }
      assert(res[RES_V_H_R].state == RES_VALID);
    }
    // An interrupting PW GPU call aborts the chain; the caller recomputes.
    assert(!hartree_fused(&h, tot, v, &e, 1));
    assert(!res_hartree.pending);
    // Complex influence functions are used without compaction.
    h.infl[5] = 1e-3;
    res_invalidate(RES_INFL);
    hartree_reference(&h, ref_tot, ref_v, &e_ref);
    assert(hartree_fused(&h, tot, v, &e, 0));
    same(ref_v, v, (size_t)nr_[0], "hartree complex influence");
    assert(res[RES_INFL].owned_bytes == 0);
    finish();
    hartree_free(&h);
  }
}

// XC input residency: the pipeline result is identical with and without the
// densities left on the device by density_rs2pw, and skips their uploads.
static int xc_call(const double *rho_g, const double *rho_r, double *vxc, double *exc) {
  return pw_gpu_xc_full_pipeline_configured_c(rho_g, rho_r, vxc, exc, map_[0], g_fine,
      dims[0], ng_[0], nmaps_, 1.0 / nr_[0], 1, 1e-10, .804, .066725,
      .2195149727645171, 1.0, 1.0, 1e-10, 0.0, 0);
}

static void chain(void) {
  for (int nmaps = 1; nmaps <= 2; nmaps++) {
    grids(nmaps);
    start(0);
    const int n = ng_[0], nr = nr_[0];
    double *r[NLEV];
    for (int l = 0; l < NLEV; l++) {
      r[l] = rarray((size_t)nr_[l]);
      for (int i = 0; i < nr_[l]; i++) r[l][i] = 0.5 + 0.2 * r[l][i];
    }
    double *rho_g = rarray(2 * (size_t)n), *rho_r = rarray((size_t)nr);
    // ---- reference (original path)
    double ref_g[2 * 64], ref_r[64], ref_vxc[64], ref_exc, ref_tot[2 * 64], ref_vh[64], e_ref;
    density_reference(r, ref_g, ref_r);
    hartree_in h = hartree_inputs();
    memcpy(h.rho_g, ref_g, sizeof(double) * 2 * (size_t)n);
    hartree_reference(&h, ref_tot, ref_vh, &e_ref);
    assert(xc_call(ref_g, ref_r, ref_vxc, &ref_exc));
    double ref_vtot[64];
    for (int i = 0; i < nr; i++) ref_vtot[i] = ref_vxc[i] * h.dvol; // pw_axpy beta
    for (int i = 0; i < nr; i++) ref_vtot[i] = ref_vtot[i] + ref_vh[i];
    const double rscale[NLEV] = {1.0, 8.0, 0.5};
    double *ref_pot[NLEV], *pot[NLEV];
    for (int l = 0; l < NLEV; l++) { ref_pot[l] = rarray((size_t)nr_[l]); pot[l] = rarray((size_t)nr_[l]); }
    potential_reference(ref_vtot, ref_pot, rscale);
    finish();

    // ---- resident chain
    start(0);
    assert(density_fused(r, rho_g, rho_r));
    if (!opt_res_lazy_density) same(ref_g, rho_g, 2 * (size_t)n, "chain rho_g");
    free(h.rho_g);
    h.rho_g = rho_g; // the host density array of the SCF
    double tot[2 * 64], vh[64], vxc[64], exc, e;
    size_t up = mock_upload_bytes;
    assert(hartree_fused(&h, tot, vh, &e, 0));
    same(ref_vh, vh, (size_t)nr, "chain v_h");
    assert(res[RES_RHO_G].hits == 1);
    up = mock_upload_bytes;
    const size_t down = mock_download_bytes;
    assert(xc_call(rho_g, rho_r, vxc, &exc));
    same(ref_vxc, vxc, (size_t)nr, "chain vxc");
    same(&ref_exc, &exc, 1, "chain exc");
    // Neither density was uploaded again (only the g-vectors, first use).
    assert(mock_upload_bytes - up == sizeof(double) * 3 * (size_t)n);
    // Consumed densities are discarded; only the potential and energy return.
    assert(mock_download_bytes - down == sizeof(double) * ((size_t)nr + 1));
    for (int id = RES_RHO_G; id <= RES_RHO_R; id++) {
      assert(res[id].state == RES_INVALID && !res[id].dev_authoritative);
    }
    assert(res[RES_RHO_R].hits == 1 && res[RES_VXC].state == RES_VALID);
    assert(pw_gpu_res_vtot_prepare_c(vh, vxc, (size_t)nr, h.dvol));
    if (opt_res_lazy_density) {
      assert(pw_gpu_res_vtot_skip_host_c());
      assert(pw_gpu_res_vtot_skip_host_c());
      same(ref_vxc, vxc, (size_t)nr, "host AXPY skipped");
    } else {
      for (int i = 0; i < nr; i++) vxc[i] = vxc[i] * h.dvol;
      for (int i = 0; i < nr; i++) vxc[i] = vxc[i] + vh[i];
      assert(pw_gpu_res_vtot_commit_c(vxc, (size_t)nr));
      same(ref_vtot, vxc, (size_t)nr, "chain vtot");
    }
    up = mock_upload_bytes;
    assert(potential_fused(vxc, pot, rscale));
    assert(mock_upload_bytes == up); // V_H + dvol*Vxc was resident
    for (int l = 0; l < NLEV; l++) same(ref_pot[l], pot[l], (size_t)nr_[l], "chain potential");
    assert(res_stats.vtot == 1 && res_stats.potential == 1 && res_stats.xc_in == 1);
    pw_gpu_res_ensure_host_c(RES_VXC, vxc);
    same(ref_vtot, vxc, (size_t)nr, "deferred host vtot");

    // A commit that disagrees with the device (e.g. an extra host term) is rejected.
    // XC consumed the previous borrowed densities; create the next iteration.
    assert(density_fused(r, rho_g, rho_r));
    assert(xc_call(rho_g, rho_r, vxc, &exc));
    same(ref_vxc, vxc, (size_t)nr, "next iteration vxc");
    same(&ref_exc, &exc, 1, "next iteration exc");
    assert(pw_gpu_res_vtot_prepare_c(vh, vxc, (size_t)nr, h.dvol));
    for (int i = 0; i < nr; i++) vxc[i] = vxc[i] * h.dvol + vh[i] + 1e-3;
    assert(!pw_gpu_res_vtot_commit_c(vxc, (size_t)nr));
    assert(res[RES_VXC].state == RES_INVALID);
    up = mock_upload_bytes;
    assert(potential_fused(vxc, pot, rscale)); // falls back to the upload
    assert(mock_upload_bytes - up == sizeof(double) * (size_t)nr);

    // Host modification of a resident density invalidates it.
    assert(density_fused(r, rho_g, rho_r));
    pw_gpu_res_ensure_host_c(RES_RHO_R, rho_r);
    rho_r[0] += 1.0;
    up = mock_upload_bytes;
    assert(xc_call(rho_g, rho_r, vxc, &exc));
    assert(mock_upload_bytes - up >= sizeof(double) * (size_t)nr);
    rho_r[0] -= 1.0;
    finish();
    for (int l = 0; l < NLEV; l++) { free(r[l]); free(ref_pot[l]); free(pot[l]); }
    free(rho_r);
    hartree_free(&h);
  }
}

// The real density FFT returns only samples. Explicit host synchronization and
// public invalidation are idempotent; reclamation preserves live results.
static void lazy_density(void) {
  for (int nmaps = 1; nmaps <= 2; nmaps++) {
    grids(nmaps);
    start(0);
    assert(opt_res_lazy_density);
    double *r[NLEV];
    for (int l = 0; l < NLEV; l++) r[l] = rarray((size_t)nr_[l]);
    double g[128] = {0}, rr[64] = {0}, ref_g[128], ref_r[64];
    density_reference(r, ref_g, ref_r);
    mock_reset_trace();
    assert(density_fused(r, g, rr));
    for (int i = 0; i < 2 * ng_[0]; i++) assert(g[i] == 0.0);
    for (int i = 0; i < nr_[0]; i++) assert(rr[i] == 0.0);
    for (size_t i = 0; i < mock_operation_count; i++) {
      if (mock_operations[i].kind == MOCK_DOWNLOAD)
        assert(mock_operations[i].buffer == res_small_dev);
    }
    assert(res[RES_RHO_G].dev_authoritative && res[RES_RHO_R].dev_authoritative);
    assert(res_samples_match(&res[RES_RHO_G], ref_g));
    assert(res_samples_match(&res[RES_RHO_R], ref_r));
    mock_reset_trace();
    double sum = -123.0;
    assert(pw_gpu_res_integrate_r_c(&sum, 0.125));
    long double reference = 0.0L;
    for (int i = 0; i < nr_[0]; i++) reference += ref_r[i];
    assert(fabsl(sum - reference * 0.125L) < 1e-12L);
    assert(mock_download_bytes == sizeof(double) && mock_upload_bytes == 0);
    // Wrong ID/pointer and invalid IDs do not materialize either slot.
    pw_gpu_res_ensure_host_c(RES_RHO_G, rr);
    pw_gpu_res_ensure_host_c(RES_NSLOTS, g);
    pw_gpu_res_ensure_host_c(-2, g);
    pw_gpu_res_ensure_host_c(-1, NULL);
    assert(res[RES_RHO_G].dev_authoritative && res[RES_RHO_R].dev_authoritative);
    mock_reset_trace();
    // Concurrent host consumers of a slice trigger only one download.
#pragma omp parallel
    pw_gpu_res_ensure_host_c(-1, rr + 1);
    assert(mock_download_bytes == sizeof(double) * (size_t)nr_[0]);
    same(rr, ref_r, (size_t)nr_[0], "lazy real sync");
    mock_reset_trace();
    pw_gpu_res_ensure_host_c(-1, rr);
    assert(mock_operation_count == 0);
    pw_gpu_res_invalidate_c();
    same(g, ref_g, 2 * (size_t)ng_[0], "lazy invalidate");
    assert(mock_download_bytes == 2 * sizeof(double) * (size_t)ng_[0]);
    // The public SCF/forces boundary must materialize both lazy densities.
    memset(g, 0, sizeof(g));
    memset(rr, 0, sizeof(rr));
    assert(density_fused(r, g, rr));
    mock_reset_trace();
    pw_gpu_res_invalidate_c();
    assert(mock_download_bytes == sizeof(double) * (2 * (size_t)ng_[0] + (size_t)nr_[0]));
    same(g, ref_g, 2 * (size_t)ng_[0], "public invalidate g");
    same(rr, ref_r, (size_t)nr_[0], "public invalidate r");
    mock_reset_trace();
    pw_gpu_res_invalidate_c();
    assert(mock_operation_count == 0);
    // A mismatched layout still needs the original data for host fallback.
    assert(density_fused(r, g, rr));
    mock_reset_trace();
    assert(!pw_gpu_res_matches_c(RES_RHO_R, rr, (size_t)nr_[0] - 1));
    assert(mock_download_bytes == sizeof(double) * (size_t)nr_[0]);
    assert(res[RES_RHO_G].dev_authoritative);
    same(rr, ref_r, (size_t)nr_[0], "lazy layout fallback");
    pw_gpu_res_invalidate_c();
    // Reclamation must preserve a new result, not the previous host values.
    r[0][0] += 0.5;
    density_reference(r, ref_g, ref_r);
    assert(density_fused(r, g, rr));
    mock_reset_trace();
    release_xc_buffer();
    assert(mock_download_bytes == sizeof(double) * (2 * (size_t)ng_[0] + (size_t)nr_[0]));
    same(g, ref_g, 2 * (size_t)ng_[0], "lazy reclaim g");
    same(rr, ref_r, (size_t)nr_[0], "lazy reclaim r");
    assert(!pw_gpu_res_integrate_r_c(&sum, 0.125));
    // CPU XC construction can use the standalone gradient after rejecting the
    // full pipeline. Its host input must be materialized at that consumer.
    double grad[3][64], grad_ref[3][64];
    pw_gpu_sfffc_grad3(ref_g, grad_ref[0], grad_ref[1], grad_ref[2], map_[0],
                       g_fine, dims[0], ng_[0], nmaps, 1);
    memset(g, 0, sizeof(g));
    memset(rr, 0, sizeof(rr));
    assert(density_fused(r, g, rr));
    mock_reset_trace();
    pw_gpu_sfffc_grad3(g, grad[0], grad[1], grad[2], map_[0],
                       g_fine, dims[0], ng_[0], nmaps, 1);
    for (int d = 0; d < 3; d++) same(grad[d], grad_ref[d], (size_t)nr_[0], "lazy XC gradient");
    assert(mock_download_bytes == sizeof(double) * (2 * (size_t)ng_[0] + 3 * (size_t)nr_[0]));
    assert(!res[RES_RHO_G].dev_authoritative && res[RES_RHO_R].dev_authoritative);
    // Disabling XC input reuse must synchronize before its host uploads.
    assert(density_fused(r, g, rr));
    opt_res_xcin = false;
    double v[64], e, vref[64], eref;
    assert(xc_call(g, rr, v, &e));
    assert(xc_call(ref_g, ref_r, vref, &eref));
    same(v, vref, (size_t)nr_[0], "lazy XC upload fallback");
    same(&e, &eref, 1, "lazy XC energy");
    finish();
    for (int l = 0; l < NLEV; l++) free(r[l]);
  }
}

// Large resident arrays exercise partial fingerprinting and three reduction
// levels without the mock's deliberately slow direct DFT.
static void integrate(void) {
  const size_t sizes[] = {1, 255, 256, 257, 17409, 65537, 200003};
  double total = 987.0;
  assert(!pw_gpu_res_integrate_r_c(&total, 1.0) && total == 987.0);
  for (size_t k = 0; k < sizeof(sizes) / sizeof(sizes[0]); k++) {
    const size_t n = sizes[k], bytes = n * sizeof(double);
    double *values = malloc(bytes), *host = calloc(n, sizeof(double));
    long double reference = 0.0L;
    for (size_t i = 0; i < n; i++) {
      values[i] = i % 4 == 0 ? 1e16 : i % 4 == 1 ? 0.25 : i % 4 == 2 ? -1e16 : 0.5;
      reference += (long double)values[i];
    }
    start(0);
    assert(pw_gpu_ensure_xc_c(4 * bytes));
    offloadMemcpyAsyncHtoD(xc_dev_hi, values, bytes, stream);
    assert(res_sample_buffer(&res[RES_RHO_R], pw_res_sample_count(n)));
    double *small = res_small(pw_res_sample_count(n));
    assert(small != NULL);
    offloadStreamSynchronize(stream);
    mock_reset_trace();
    res_register_lazy(RES_RHO_R, host, n, 1, xc_dev_hi, small);
    offloadStreamSynchronize(stream);
    assert(mock_download_bytes == pw_res_sample_count(n) * sizeof(double));
    assert(res_samples_match(&res[RES_RHO_R], values));
    mock_reset_trace();
    assert(pw_gpu_res_integrate_r_c(&total, -0.125));
    assert(fabsl((long double)total + reference * 0.125L) < 1e-12L);
    assert(mock_download_bytes == sizeof(double) && mock_upload_bytes == 0);
    assert(host[0] == 0.0 && res[RES_RHO_R].dev_authoritative);
    assert(!pw_gpu_res_matches_c(RES_RHO_R, values, n));
    assert(pw_gpu_res_matches_c(RES_RHO_R, host, n));
    assert(!pw_gpu_res_matches_c(RES_NSLOTS, host, n));
    assert(!pw_gpu_res_integrate_r_c(NULL, 1.0));
    assert(!pw_gpu_res_integrate_r_c(&total, NAN));
    // Exact scratch budget, then one byte short: leave output unchanged and
    // keep the density available for a CPU fallback with no allocation.
    res_free_small();
    max_gpu_memory_bytes = mock_limit = current_gpu_allocated_bytes + PW_GPU_MIB - 1;
    total = 987.0;
    assert(!pw_gpu_res_integrate_r_c(&total, 1.0) && total == 987.0);
    assert(res[RES_RHO_R].dev_authoritative);
    max_gpu_memory_bytes = mock_limit = current_gpu_allocated_bytes + PW_GPU_MIB;
    assert(pw_gpu_res_integrate_r_c(&total, -0.125));
    assert(fabsl((long double)total + reference * 0.125L) < 1e-12L);
    pw_gpu_res_ensure_host_c(RES_RHO_R, host);
    same(host, values, n, "reduction preserved rho");
    host[0] += 16.0;
    total = 987.0;
    assert(!pw_gpu_res_integrate_r_c(&total, 1.0) && total == 987.0);
    finish();
    free(values); free(host);
  }
}

static void lazy_vtot(void) {
  grids(1);
  start(0);
  double rho_g[128] = {0}, rho_r[64], vxc[64], vh[64], exc;
  double expected[64];
  double *out[NLEV], *ref[NLEV];
  const double scales[NLEV] = {1.0, 8.0, 0.5};
  for (int l = 0; l < NLEV; l++) {
    out[l] = rarray((size_t)nr_[l]);
    ref[l] = rarray((size_t)nr_[l]);
  }
  for (int i = 0; i < nr_[0]; i++) { rho_r[i] = 0.5; vh[i] = rnd(); }
  assert(!pw_gpu_res_vtot_skip_host_c());
  for (int pass = 0; pass < 3; pass++) {
    assert(xc_call(rho_g, rho_r, vxc, &exc));
    for (int i = 0; i < nr_[0]; i++) expected[i] = 0.125 * vxc[i] + vh[i];
    potential_reference(expected, ref, scales);
    assert(pw_gpu_res_vtot_prepare_c(vh, vxc, (size_t)nr_[0], 0.125));
    assert(pw_gpu_res_vtot_skip_host_c());
    if (pass == 0) {
      opt_res_pot = false; // failed device attempt must leave a usable host input
      assert(!potential_fused(vxc, out, scales));
      opt_res_pot = true;
      same(vxc, expected, (size_t)nr_[0], "lazy potential fallback");
    } else if (pass == 1) {
      // The auxiliary output can alias the original potential.
      double *saved = out[0];
      out[0] = vxc;
      assert(potential_fused(vxc, out, scales));
      for (int l = 0; l < NLEV; l++) same(out[l], ref[l], (size_t)nr_[l], "lazy potential alias");
      assert(!res[RES_VXC].dev_authoritative);
      out[0] = saved;
    } else {
      // Lowering a live limit releases scratch only after preserving Vtot.
      pw_gpu_set_memory_limit(1);
      same(vxc, expected, (size_t)nr_[0], "lazy potential memory pressure");
    }
  }
  finish();
  for (int l = 0; l < NLEV; l++) { free(out[l]); free(ref[l]); }
}

// Memory: optional slots are skipped under a budget or without free memory,
// evicted for mandatory scratch, and the results do not change.
static void memory(void) {
  grids(1);
  start(0);
  hartree_in h = hartree_inputs();
  double ref_tot[2 * 64], tot[2 * 64], ref_v[64], v[64], e_ref, e;
  hartree_reference(&h, ref_tot, ref_v, &e_ref);
  finish();
  // No free memory beyond the reserve: no owned slot.
  start(0);
  mock_free_memory_value = res_reserve_bytes;
  assert(hartree_fused(&h, tot, v, &e, 0));
  same(ref_v, v, (size_t)nr_[0], "no free memory");
  for (int i = RES_V_H_R; i < RES_NSLOTS; i++) assert(res[i].owned_bytes == 0);
  mock_free_memory_value = (size_t)1 << 40;
  finish();
  // Budget exactly for the mandatory scratch: slots are skipped, no failure.
  start(0);
  assert(hartree_fused(&h, tot, v, &e, 0));
  const size_t with_slots = current_gpu_allocated_bytes;
  size_t owned = 0;
  for (int i = RES_V_H_R; i < RES_NSLOTS; i++) owned += res[i].owned_bytes;
  assert(owned > 0);
  finish();
  start(0);
  max_gpu_memory_bytes = mock_limit = with_slots - owned;
  assert(hartree_fused(&h, tot, v, &e, 0));
  same(ref_v, v, (size_t)nr_[0], "budget");
  for (int i = RES_V_H_R; i < RES_NSLOTS; i++) assert(res[i].owned_bytes == 0);
  accounting();
  finish();
  // Mandatory scratch evicts the owned slots first.
  start(0);
  assert(hartree_fused(&h, tot, v, &e, 0));
  max_gpu_memory_bytes = mock_limit = current_gpu_allocated_bytes;
  assert(pw_gpu_ensure_gbuf_c(allocated_gbuf_size + res[RES_INFL].owned_bytes));
  assert(res[RES_INFL].owned_bytes == 0);
  accounting();
  finish();
  hartree_free(&h);
}

// Reuse discards dead borrowed densities without downloads; memory pressure
// must still preserve live results before releasing their XC grids.
static void borrowed(void) {
  grids(1);
  start(0);
  double *r[NLEV];
  for (int l = 0; l < NLEV; l++) r[l] = rarray((size_t)nr_[l]);
  double *rho_g = rarray(2 * (size_t)ng_[0]), *rho_r = rarray((size_t)nr_[0]);
  double ref_g[128], ref_r[64], host_g[128], host_r[64];
  density_reference(r, ref_g, ref_r);
  assert(density_fused(r, rho_g, rho_r));
  memcpy(host_g, rho_g, 2 * sizeof(double) * (size_t)ng_[0]);
  memcpy(host_r, rho_r, sizeof(double) * (size_t)nr_[0]);
  const unsigned long epoch = xc_epoch;
  mock_reset_trace();
  res_invalidate_borrowed();
  assert(xc_epoch != epoch);
  for (int id = 0; id <= RES_VXC; id++) {
    assert(res[id].state == RES_INVALID && !res[id].dev_authoritative && !res[id].vtot);
  }
  // Discarded slots must not be resurrected or downloaded by later host access.
  pw_gpu_res_ensure_host_c(RES_RHO_G, rho_g);
  pw_gpu_res_ensure_host_c(RES_RHO_R, rho_r);
  pw_gpu_res_invalidate_c();
  assert(mock_operation_count == 0);
  same(host_g, rho_g, 2 * (size_t)ng_[0], "discard leaves host g unchanged");
  same(host_r, rho_r, (size_t)nr_[0], "discard leaves host r unchanged");
  assert(density_fused(r, rho_g, rho_r));
  max_gpu_memory_bytes = mock_limit = current_gpu_allocated_bytes;
  assert(pw_gpu_ensure_gbuf_c(allocated_gbuf_size + 64));
  assert(xc_dev == NULL);
  assert(res_lookup(RES_RHO_G, rho_g, (size_t)ng_[0], false) == NULL);
  same(ref_g, rho_g, 2 * (size_t)ng_[0], "reclaim preserves g");
  same(ref_r, rho_r, (size_t)nr_[0], "reclaim preserves r");
  accounting();
  finish();
  for (int l = 0; l < NLEV; l++) free(r[l]);
  free(rho_g); free(rho_r);
}

// Sampling: positions cover the array, the prefix and every bucket.
static void samples(void) {
  const size_t sizes[] = {1, 5, 1024, 1025, 17407, 17408, 17409, 100000, 66430125};
  for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
    const size_t n = sizes[s], count = pw_res_sample_count(n);
    assert(count == (n < 17408 ? n : 17408));
    size_t last = 0;
    for (size_t k = 0; k < count; k++) {
      const size_t e = pw_res_sample_index(k, n);
      assert(e < n && (k == 0 || e > last));
      last = e;
    }
  }
}

// CP2K_PW_GPU_RES_VERIFY catches what sampling cannot: an unsampled change.
static void verify(void) {
  grids(1);
  start(0);
  double *r[NLEV];
  for (int l = 0; l < NLEV; l++) r[l] = rarray((size_t)nr_[l]);
  double *rho_g = rarray(2 * (size_t)ng_[0]), *rho_r = rarray((size_t)nr_[0]);
  assert(density_fused(r, rho_g, rho_r));
  opt_res_verify = true;
  assert(res_lookup(RES_RHO_R, rho_r, (size_t)nr_[0], false) != NULL);
  assert(density_fused(r, rho_g, rho_r));
  // All elements are sampled in small arrays; emulate a missed position by
  // corrupting the device copy instead.
  double bad = 7.0;
  offloadMemcpyAsyncHtoD(res[RES_RHO_R].dev + 3, &bad, sizeof(double), stream);
  offloadStreamSynchronize(stream);
  assert(res_lookup(RES_RHO_R, rho_r, (size_t)nr_[0], false) == NULL);
  opt_res_verify = false;
  finish();
  for (int l = 0; l < NLEV; l++) free(r[l]);
  free(rho_g); free(rho_r);
}

int main(int argc, char **argv) {
  assert(argc >= 2);
  if (!strcmp(argv[1], "density")) density();
  else if (!strcmp(argv[1], "density_rejects")) density_rejects();
  else if (!strcmp(argv[1], "potential")) potential();
  else if (!strcmp(argv[1], "hartree")) hartree();
  else if (!strcmp(argv[1], "chain")) chain();
  else if (!strcmp(argv[1], "memory")) memory();
  else if (!strcmp(argv[1], "borrowed")) borrowed();
  else if (!strcmp(argv[1], "samples")) samples();
  else if (!strcmp(argv[1], "verify")) verify();
  else if (!strcmp(argv[1], "lazy_density")) lazy_density();
  else if (!strcmp(argv[1], "integrate")) integrate();
  else if (!strcmp(argv[1], "lazy_vtot")) lazy_vtot();
  else assert(!"unknown test");
  for (int l = 0; l < NLEV; l++) { free(map_[l]); free(gidx_[l]); }
  free(g_fine); free(gsq_fine);
  puts("PASS");
  return 0;
}
