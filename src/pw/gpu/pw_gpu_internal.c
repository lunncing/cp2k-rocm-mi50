/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/
#include "../../offload/offload_runtime.h"

/*******************************************************************************
 * \brief Real part of the complex inner product, without temporary arrays.
 *        Complex doubles are stored as interleaved real and imaginary parts.
 *        This CPU operation is also available in builds without PW offload.
 *        n_complex must be in [0, INT_MAX / 2] for the BLAS integer count.
 ******************************************************************************/
double pw_cpu_dot_complex_c(const void *a, const void *b, int n_complex) {
  int n_double = 2 * n_complex;
  extern double ddot_(const int *n, const double *x, const int *incx,
                      const double *y, const int *incy);
  const int inc = 1;
  return ddot_(&n_double, (const double *)a, &inc, (const double *)b, &inc);
}

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)

#include "../../offload/offload_fft.h"
#include "../../offload/offload_library.h"
#include "pw_gpu_kernels.h"

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pw_gpu_host.h"
#include "grid_passthrough.h"

/*******************************************************************************
 * \brief Static variables for retaining objects that are expensive to create.
 * \author Ole Schuett
 ******************************************************************************/
typedef struct {
  int key[4];
  offload_fftHandle *plan;
} cache_entry;

#define PW_GPU_CACHE_SIZE 32
static cache_entry cache[PW_GPU_CACHE_SIZE];
static int cache_oldest_entry = 0; // used for LRU eviction

static double *buffer_dev_1, *buffer_dev_2;
static double *buffer_dev_1_b = NULL, *buffer_dev_2_b = NULL;
static int *ghatmap_dev;
static size_t allocated_buffer_size, allocated_map_size;
static size_t allocated_buffer_size_b;

static offloadStream_t stream; // All kernels and cached FFT plans use this stream.
static offloadStream_t stream_copy;
static offloadEvent_t event_compute_done[2], event_copy_done[2];
static bool is_initialized = false;

/*******************************************************************************
 * \brief Device-resident per-grid data (g_hatmap, g-vectors).
 *
 *        A pw_grid is identified by its id_nr, which is unique for the whole
 *        run (pw_grids.F: grid_tag is incremented for every created grid and
 *        never reused). g_hatmap is computed once in pw_grid_create_ghatmap and
 *        never changes afterwards. The g-vectors can change in place via
 *        pw_grid_change (cell change); pw_grids.F therefore calls
 *        pw_gpu_grid_invalidate() from pw_grid_change and pw_grid_release.
 *        As an additional guard, an entry is only reused if host pointer,
 *        ngpts and nmaps match as well.
 ******************************************************************************/
typedef struct {
  int grid_id; // 0 marks an empty slot, valid ids are >= 1
  int ngpts;
  int nmaps;
  const int *map_host;
  const double *g_host;
  int *map_dev;
  double *g_dev; // only allocated when a fused derivative path needs it
  const int *gidx_host; // reference-grid index (one-based), multigrid paths
  int *gidx_dev;
  int gidx_bound;       // validated upper bound of the gidx entries
  const int *gidx_natural_map; // NULL: sorted G, otherwise target FFT box
  unsigned long last_use;
} grid_cache_entry;

#define PW_GPU_GRID_CACHE_SIZE 8
static grid_cache_entry grid_cache[PW_GPU_GRID_CACHE_SIZE];
static unsigned long grid_cache_clock = 0;
// The map being extended with g-vectors must survive memory reclamation.
static grid_cache_entry *protected_grid_entry = NULL;

// Compact (ngpts complex) device scratch for the fused paths.
static double *gbuf_dev;
static size_t allocated_gbuf_size;

// Four real grids: three gradients (later divergence inputs) and dE/drho.
// The two shared complex buffers hold rho/norm and FFT/e0/dE/dnorm by lifetime.
// The grids are two allocations (grids 0-1 and 2-3) so that the resident
// density can be downloaded from one while the other is written.
static double *xc_dev, *xc_dev_hi;
static size_t allocated_xc_size;
static bool protected_xc_buffer = false;

/*******************************************************************************
 * \brief Device residency of PW arrays between calls (density -> Hartree ->
 *        XC -> potential). A slot mirrors one host array. Lazy results are
 *        authoritative on device until a host consumer requests them. Host
 *        consumers must synchronize before reading or changing the array.
 *        Synchronized copies are checked by sampled bit pattern on reuse.
 *        Borrowed slots live in the XC grids and are tied to xc_epoch, which
 *        changes whenever those grids are reallocated, freed or overwritten.
 *        Owned slots are optional allocations, evicted before any mandatory
 *        scratch and only created when they fit the budget and free memory.
 ******************************************************************************/
enum {
  RES_RHO_G = 0, // borrowed: xc_dev[0, 2 ngpts), or a full natural FFT box
  RES_RHO_R,     // borrowed: xc_dev_hi[0, nr)
  RES_VXC,       // borrowed: xc_dev[0, nr); Vxc, later dvol*Vxc + V_H
  RES_V_H_R,     // owned
  RES_RHO_CORE,  // owned
  RES_INFL,      // owned, compact real influence when its imaginary part is zero
  RES_NSLOTS
};
enum { RES_INVALID = 0, RES_VALID, RES_PENDING };
typedef struct {
  const void *host;
  size_t n;           // elements
  int width;          // doubles per element on the device
  int host_width;     // doubles per element on the host (INFL: 2 vs 1)
  double *dev;
  size_t owned_bytes; // 0 for borrowed slots
  unsigned long epoch;
  int state;
  bool vtot;          // RES_VXC holds dvol*Vxc + V_H instead of Vxc
  bool natural_g;     // G-space storage uses res_natural's FFT box layout
  bool dev_authoritative; // host must be synchronized before access
  double *samples;    // host copy of the fingerprint values
  size_t sample_capacity;
  unsigned long hits, misses;
} res_slot;
static res_slot res[RES_NSLOTS];
static const char *const res_names[RES_NSLOTS] = {"RHO_G", "RHO_R", "VXC",
                                                  "V_H_R", "RHO_CORE", "INFL"};
static unsigned long xc_epoch = 1;      // borrowed slot generation
static unsigned long scratch_epoch = 1; // changes with every shared scratch use
static unsigned res_pinned = 0;         // owned slots in use (never evicted)
static double *res_small_dev = NULL;    // small scratch (samples, DDAPC data)
static size_t res_small_bytes = 0;
static bool opt_res = true, opt_res_density = true, opt_res_hartree = true;
static bool opt_res_xcin = true, opt_res_pot = true, opt_res_verify = false;
static bool opt_res_stats = true;
static bool opt_res_lazy_density = true;
static bool opt_res_natural_g = true;
static bool opt_res_lazy_pot = true; // lazy Vxc and V_H (inside the granted window)
// Lazy (device-authoritative) results are only created inside a window that
// the SCF driver grants for setups without host readers between a result and
// its device consumer (pw_gpu_res_set_lazy_c). The density token marks the
// one density computation of the SCF iteration that may stay on the device.
static bool res_lazy_grant = false, res_density_token = false;
static size_t res_reserve_bytes = (size_t)1536 * 1024 * 1024;
static struct {
  unsigned long density, density_fail, hartree, hartree_fail, potential,
      potential_fail, vtot, vtot_fail, xc_in;
  unsigned long lazy_density, lazy_vxc, lazy_vhr, discards, syncs;
  size_t upload_saved, download_saved, sync_bytes;
} res_stats;
// Hartree/DDAPC chain state between its Fortran-driven steps.
static struct {
  bool pending;
  unsigned long epoch;
  size_t ngpts;
  bool natural_g;
  int npts[3];
  int g0_done, modify_done;
} res_hartree;
// The local full-box map is host-owned and immutable until grid invalidation.
// No full target map is kept on the device by density/Hartree. Bounded staging
// also makes a lazy natural density synchronizable without a new allocation.
static struct {
  int grid_id, npts[3];
  size_t n;
  const int *map;
} res_natural;
static double *res_natural_dev = NULL;
static size_t res_natural_capacity = 0;
#define PW_GPU_RES_NATURAL_CHUNK 65536
static void res_natural_forget(void);
static void res_evict_all(void);
static bool res_evict_one(void);
static void res_invalidate_borrowed(void);
static void res_invalidate(const int id);
static void res_print_stats(void);
void pw_gpu_res_ensure_host_c(const int id, void *host_ptr);

// Runtime switches, read once in pw_gpu_init().
static bool opt_map_cache = true; // CP2K_PW_GPU_MAP_CACHE
static bool opt_fused_xc = true;  // CP2K_PW_GPU_FUSED_XC

// Counts all explicit device allocations in this module. FFT-library internal
// workspaces are opaque and are not included in this budget.
static size_t max_gpu_memory_bytes = 0;
static int host_fallback_enabled = 1;
static size_t current_gpu_allocated_bytes = 0;
static size_t peak_gpu_allocated_bytes = 0;
static bool memory_environment_read = false;
static bool memory_limit_from_environment = false;
static bool host_fallback_from_environment = false;
static bool fallback_notice_printed = false;

#define PW_GPU_MIB ((size_t)1024 * 1024)

// Accept only complete, nonnegative decimal integers and prevent conversion
// overflow before multiplying MiB. Invalid overrides do not hide valid defaults.
static bool parse_unsigned(const char *s, const size_t maximum, size_t *value) {
  if (s == NULL) {
    return false;
  }
  while (isspace((unsigned char)*s)) {
    s++;
  }
  if (!isdigit((unsigned char)*s)) {
    return false;
  }
  errno = 0;
  char *end;
  const unsigned long long parsed = strtoull(s, &end, 10);
  if (errno == ERANGE || parsed > maximum) {
    return false;
  }
  while (isspace((unsigned char)*end)) {
    end++;
  }
  if (*end != '\0') {
    return false;
  }
  *value = (size_t)parsed;
  return true;
}

/*******************************************************************************
 * \brief Reads a boolean environment switch ("0", "no", "false", "off" = off).
 ******************************************************************************/
static bool env_flag(const char *name, const bool default_value) {
  const char *s = getenv(name);
  if (s == NULL || s[0] == '\0') {
    return default_value;
  }
  if (s[0] == '0' || s[0] == 'n' || s[0] == 'N' || s[0] == 'f' ||
      s[0] == 'F') {
    return false;
  }
  if ((s[0] == 'o' || s[0] == 'O') && (s[1] == 'f' || s[1] == 'F')) {
    return false; // "off"
  }
  return true;
}

static void read_memory_environment(void) {
  if (memory_environment_read) {
    return;
  }
  memory_environment_read = true;
  const char *rank_names[] = {"OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID",
                              "MPI_LOCALRANKID"};
  size_t rank = 0, mb;
  for (size_t i = 0; i < sizeof(rank_names) / sizeof(rank_names[0]); i++) {
    if (parse_unsigned(getenv(rank_names[i]), INT_MAX, &rank)) {
      break;
    }
  }
  char name[80];
  snprintf(name, sizeof(name), "CP2K_GPU_MEM_LIMIT_MB_RANK%zu", rank);
  memory_limit_from_environment =
      parse_unsigned(getenv(name), SIZE_MAX / PW_GPU_MIB, &mb) ||
      parse_unsigned(getenv("CP2K_GPU_MEM_LIMIT_MB"), SIZE_MAX / PW_GPU_MIB, &mb);
  if (memory_limit_from_environment) {
    max_gpu_memory_bytes = mb * PW_GPU_MIB;
  }
  const char *fallback = getenv("CP2K_GPU_HOST_FALLBACK");
  host_fallback_from_environment = (fallback != NULL && fallback[0] != '\0');
  if (host_fallback_from_environment) {
    host_fallback_enabled = env_flag("CP2K_GPU_HOST_FALLBACK", true);
  }
}

static void device_malloc(void **ptr, const size_t bytes) {
  assert(bytes <= SIZE_MAX - current_gpu_allocated_bytes);
  offloadMalloc(ptr, bytes);
  current_gpu_allocated_bytes += bytes;
  if (current_gpu_allocated_bytes > peak_gpu_allocated_bytes) {
    peak_gpu_allocated_bytes = current_gpu_allocated_bytes;
  }
}

static void device_free(void *ptr, const size_t bytes) {
  if (ptr != NULL) {
    assert(bytes <= current_gpu_allocated_bytes);
    offloadFree(ptr);
    current_gpu_allocated_bytes -= bytes;
  }
}

// Optional scratch must never force a transformation onto the host. Both
// streams can still reference it when resizing or reclaiming device memory.
static void release_double_buffers(void) {
  if (allocated_buffer_size_b == 0) {
    return;
  }
  offloadStreamSynchronize(stream);
  offloadStreamSynchronize(stream_copy);
  device_free(buffer_dev_1_b, allocated_buffer_size_b / 2);
  device_free(buffer_dev_2_b, allocated_buffer_size_b);
  buffer_dev_1_b = buffer_dev_2_b = NULL;
  allocated_buffer_size_b = 0;
}

/*******************************************************************************
 * \brief Frees the device memory of a grid cache entry and empties the slot.
 ******************************************************************************/
static void grid_cache_clear_entry(grid_cache_entry *e) {
  if (e->map_dev != NULL || e->g_dev != NULL || e->gidx_dev != NULL) {
    // Uploads are asynchronous, including entries reclaimed during preparation.
    offloadStreamSynchronize(stream);
  }
  device_free(e->map_dev, sizeof(int) * (size_t)e->ngpts * (size_t)e->nmaps);
  device_free(e->g_dev, 3 * sizeof(double) * (size_t)e->ngpts);
  device_free(e->gidx_dev, sizeof(int) * (size_t)e->ngpts);
  memset(e, 0, sizeof(grid_cache_entry));
}

// Size of the first of the two XC grid allocations, in whole doubles.
static size_t xc_lo_bytes(const size_t total) {
  return (total / 2 + sizeof(double) - 1) / sizeof(double) * sizeof(double);
}

static void release_xc_buffer(void) {
  if (xc_dev == NULL) return;
  // Reclamation may evict live results, unlike reuse after their last consumer.
  for (int id = 0; id <= RES_VXC; id++) {
    pw_gpu_res_ensure_host_c(id, (void *)res[id].host);
  }
  res_invalidate_borrowed();
  offloadStreamSynchronize(stream);
  offloadStreamSynchronize(stream_copy);
  const size_t lo = xc_lo_bytes(allocated_xc_size);
  device_free(xc_dev, lo);
  device_free(xc_dev_hi, allocated_xc_size - lo);
  xc_dev = xc_dev_hi = NULL;
  allocated_xc_size = 0;
}

static int check_and_reclaim_memory(const size_t required_bytes) {
  read_memory_environment();
  if (max_gpu_memory_bytes == 0) {
    return required_bytes <= SIZE_MAX - current_gpu_allocated_bytes;
  }
  while (current_gpu_allocated_bytes > max_gpu_memory_bytes ||
         required_bytes > max_gpu_memory_bytes - current_gpu_allocated_bytes) {
    if (allocated_buffer_size_b != 0) {
      release_double_buffers();
      continue;
    }
    if (res_evict_one()) {
      continue;
    }
    if (xc_dev != NULL && !protected_xc_buffer) {
      release_xc_buffer();
      continue;
    }
    grid_cache_entry *oldest = NULL;
    for (int i = 0; i < PW_GPU_GRID_CACHE_SIZE; i++) {
      grid_cache_entry *e = &grid_cache[i];
      if (e->grid_id != 0 && e != protected_grid_entry &&
          (oldest == NULL || e->last_use < oldest->last_use)) {
        oldest = e;
      }
    }
    if (oldest == NULL) {
      return 0;
    }
    grid_cache_clear_entry(oldest);
  }
  return 1;
}

static int memory_limit_exceeded(void) {
  if (is_initialized) {
    offloadStreamSynchronize(stream);
    offloadStreamSynchronize(stream_copy);
  }
  if (!host_fallback_enabled) {
    fprintf(stderr, "[PW_GPU] Memory limit reached (%zu MB); host fallback is disabled.\n",
            max_gpu_memory_bytes / PW_GPU_MIB);
    abort();
  }
  if (!fallback_notice_printed) {
    fprintf(stderr, "[PW_GPU] Memory limit reached (%zu MB), falling back to host.\n",
            max_gpu_memory_bytes / PW_GPU_MIB);
    fallback_notice_printed = true;
  }
  return 0;
}

static void release_work_buffers(void) {
  offloadStreamSynchronize(stream);
  offloadStreamSynchronize(stream_copy);
  release_double_buffers();
  release_xc_buffer();
  res_evict_all();
  scratch_epoch++;
  res_hartree.pending = false;
  device_free(buffer_dev_1, allocated_buffer_size);
  device_free(buffer_dev_2, allocated_buffer_size);
  device_free(ghatmap_dev, allocated_map_size);
  device_free(gbuf_dev, allocated_gbuf_size);
  buffer_dev_1 = buffer_dev_2 = gbuf_dev = NULL;
  ghatmap_dev = NULL;
  allocated_buffer_size = allocated_map_size = allocated_gbuf_size = 0;
}

void pw_gpu_set_memory_limit(const int limit_mb) {
  assert(omp_get_num_threads() == 1);
  read_memory_environment();
  if (!memory_limit_from_environment) {
    if (limit_mb < 0 || (size_t)limit_mb > SIZE_MAX / PW_GPU_MIB) {
      fprintf(stderr, "[PW_GPU] Invalid GPU memory limit.\n");
      abort();
    }
    max_gpu_memory_bytes = (size_t)limit_mb * PW_GPU_MIB;
  }
  if (is_initialized) {
    offload_activate_chosen_device();
    if (!check_and_reclaim_memory(0)) {
      // Calls are synchronous at the public boundary, so idle scratch can be
      // released when a caller lowers the limit between transformations.
      release_work_buffers();
    }
  }
}

void pw_gpu_set_host_fallback(const int enable) {
  read_memory_environment();
  if (!host_fallback_from_environment) {
    host_fallback_enabled = (enable != 0);
  }
}

void pw_gpu_get_allocated_bytes(size_t *allocated_bytes, size_t *peak_bytes) {
  read_memory_environment();
  *allocated_bytes = current_gpu_allocated_bytes;
  *peak_bytes = peak_gpu_allocated_bytes;
}

/*******************************************************************************
 * \brief Initializes the pw_gpu library.
 * \author Ole Schuett
 ******************************************************************************/
void pw_gpu_init(void) {
  assert(omp_get_num_threads() == 1);
  read_memory_environment();
  if (is_initialized) {
    // fprintf(stderr, "Error: pw_gpu was already initialized.\n");
    // TODO abort();
    return;
  }
  memset(cache, 0, sizeof(cache_entry) * PW_GPU_CACHE_SIZE);
  cache_oldest_entry = 0;

  // Lazy allocation keeps initialization within even the smallest budget.
  allocated_buffer_size = allocated_map_size = allocated_gbuf_size = 0;
  allocated_buffer_size_b = 0;
  allocated_xc_size = 0;
  current_gpu_allocated_bytes = peak_gpu_allocated_bytes = 0;
  offload_activate_chosen_device();

  memset(grid_cache, 0, sizeof(grid_cache_entry) * PW_GPU_GRID_CACHE_SIZE);
  grid_cache_clock = 0;
  opt_map_cache = env_flag("CP2K_PW_GPU_MAP_CACHE", true);
  opt_fused_xc = env_flag("CP2K_PW_GPU_FUSED_XC", true);
  opt_res = env_flag("CP2K_PW_GPU_RES", true);
  opt_res_density = opt_res && env_flag("CP2K_PW_GPU_RES_DENSITY", true);
  opt_res_hartree = opt_res && env_flag("CP2K_PW_GPU_RES_HARTREE", true);
  opt_res_xcin = opt_res && env_flag("CP2K_PW_GPU_RES_XCIN", true);
  opt_res_pot = opt_res && env_flag("CP2K_PW_GPU_RES_POT", true);
  opt_res_verify = opt_res && env_flag("CP2K_PW_GPU_RES_VERIFY", false);
  opt_res_lazy_density = opt_res && env_flag("CP2K_PW_GPU_RES_LAZY_DENSITY", true);
  opt_res_natural_g = opt_res && env_flag("CP2K_PW_GPU_RES_NATURAL_G", true);
  opt_res_lazy_pot = opt_res && env_flag("CP2K_PW_GPU_RES_LAZY_POT", true);
  opt_res_stats = opt_res && env_flag("CP2K_PW_GPU_RES_STATS", true);
  {
    size_t mb;
    if (parse_unsigned(getenv("CP2K_PW_GPU_RES_RESERVE_MB"), SIZE_MAX / PW_GPU_MIB, &mb)) {
      res_reserve_bytes = mb * PW_GPU_MIB;
    }
  }
  memset(res, 0, sizeof(res));
  memset(&res_stats, 0, sizeof(res_stats));
  res_lazy_grant = res_density_token = false;
  memset(&res_hartree, 0, sizeof(res_hartree));
  memset(&res_natural, 0, sizeof(res_natural));
  res_pinned = 0;
  res_small_dev = NULL;
  res_small_bytes = 0;
  xc_epoch++;
  scratch_epoch++;

  offloadStreamCreate(&stream);
  offloadStreamCreate(&stream_copy);
  for (int i = 0; i < 2; i++) {
    offloadEventCreate(&event_compute_done[i]);
    offloadEventCreate(&event_copy_done[i]);
  }
  is_initialized = true;
}

/*******************************************************************************
 * \brief Returns the runtime switches (bit 0: map cache, bit 1: fused XC,
 *        bits 2-5: residency of density, Hartree, XC input, potential).
 ******************************************************************************/
int pw_gpu_get_options(void) {
  read_memory_environment();
  if (!is_initialized) {
    return 0;
  }
  return (opt_map_cache ? 1 : 0) + (opt_fused_xc ? 2 : 0) +
         (opt_res_density ? 4 : 0) + (opt_res_hartree ? 8 : 0) +
         (opt_res_xcin ? 16 : 0) + (opt_res_pot ? 32 : 0);
}

/*******************************************************************************
 * \brief Returns 1 if the fused XC gradient/divergence/filter paths are on.
 ******************************************************************************/
int pw_gpu_fused_xc_enabled(void) {
  read_memory_environment();
  return (is_initialized && opt_fused_xc) ? 1 : 0;
}

/*******************************************************************************
 * \brief Drops all device data cached for the given grid. Has to be called
 *        whenever the g-vectors of a grid change or the grid is released.
 ******************************************************************************/
void pw_gpu_grid_invalidate(const int grid_id) {
  if (!is_initialized || grid_id <= 0) {
    return;
  }
#pragma omp critical(pw_gpu_grid_cache)
  {
    if (res_natural.grid_id == grid_id) res_natural_forget();
    bool found = false;
    for (int i = 0; i < PW_GPU_GRID_CACHE_SIZE; i++) {
      if (grid_cache[i].grid_id == grid_id) {
        found = true;
      }
    }
    if (found) {
      offload_activate_chosen_device();
      for (int i = 0; i < PW_GPU_GRID_CACHE_SIZE; i++) {
        if (grid_cache[i].grid_id == grid_id) {
          grid_cache_clear_entry(&grid_cache[i]);
        }
      }
    }
  }
}

/*******************************************************************************
 * \brief Releases resources held by the pw_gpu library.
 * \author Ole Schuett
 ******************************************************************************/
void pw_gpu_finalize(void) {
  assert(omp_get_num_threads() == 1);
  if (!is_initialized) {
    // fprintf(stderr, "Error: pw_gpu is not initialized.\n");
    // TODO abort();
    return;
  }
  offload_activate_chosen_device();
  offloadStreamSynchronize(stream);
  offloadStreamSynchronize(stream_copy);
  for (int i = 0; i < PW_GPU_CACHE_SIZE; i++) {
    if (cache[i].plan != NULL) {
      offload_fftDestroy(*cache[i].plan);
      free(cache[i].plan);
    }
  }
  for (int i = 0; i < PW_GPU_GRID_CACHE_SIZE; i++) {
    grid_cache_clear_entry(&grid_cache[i]);
  }
  // The host arrays may already be released here: forget live results instead
  // of writing them back. The SCF exit and force boundaries leave none.
  int live = 0;
  for (int i = 0; i < RES_NSLOTS; i++) {
    live += res[i].dev_authoritative ? 1 : 0;
    res_invalidate(i);
  }
  if (live != 0) {
    fprintf(stderr, "[PW_GPU] residency: %d device-only result(s) dropped at finalization.\n",
            live);
  }
  res_lazy_grant = res_density_token = false;
  release_work_buffers();
  res_print_stats();
  for (int i = 0; i < RES_NSLOTS; i++) {
    free(res[i].samples);
    res[i].samples = NULL;
    res[i].sample_capacity = 0;
  }
  assert(current_gpu_allocated_bytes == 0);
  for (int i = 0; i < 2; i++) {
    offloadEventDestroy(event_compute_done[i]);
    offloadEventDestroy(event_copy_done[i]);
  }
  offloadStreamDestroy(stream_copy);
  offloadStreamDestroy(stream);
  is_initialized = false;
}

/*******************************************************************************
 * \brief Checks size of device buffers and re-allocates them if necessary.
 * \author Ole Schuett
 ******************************************************************************/
static int pw_gpu_ensure_buffer_c(const size_t requested_buffer_size,
                                  const size_t requested_map_size,
                                  const bool double_buffer) {
  assert(is_initialized);
  scratch_epoch++; // every user of the shared scratch passes through here
  const size_t buffer_growth = requested_buffer_size > allocated_buffer_size
                                  ? requested_buffer_size - allocated_buffer_size
                                  : 0;
  const size_t map_growth = requested_map_size > allocated_map_size
                               ? requested_map_size - allocated_map_size
                               : 0;
  if (buffer_growth != 0) {
    release_double_buffers();
  }
  if (buffer_growth > (SIZE_MAX - map_growth) / 2 ||
      !check_and_reclaim_memory(2 * buffer_growth + map_growth)) {
    return memory_limit_exceeded();
  }
  if (buffer_growth != 0 || map_growth != 0) {
    offloadStreamSynchronize(stream);
  }
  if (requested_buffer_size > allocated_buffer_size) {
    device_free(buffer_dev_1, allocated_buffer_size);
    device_free(buffer_dev_2, allocated_buffer_size);
    device_malloc((void **)&buffer_dev_1, requested_buffer_size);
    device_malloc((void **)&buffer_dev_2, requested_buffer_size);
    allocated_buffer_size = requested_buffer_size;
  }
  if (requested_map_size > allocated_map_size) {
    device_free(ghatmap_dev, allocated_map_size);
    device_malloc((void **)&ghatmap_dev, requested_map_size);
    allocated_map_size = requested_map_size;
  }
  // Request these only after mandatory scratch and the active grid are ready.
  // The caller protects that grid from LRU eviction during this reservation.
  if (double_buffer && requested_buffer_size > allocated_buffer_size_b) {
    release_double_buffers();
    const size_t real_size = requested_buffer_size / 2;
    if (real_size != 0 && requested_buffer_size <= SIZE_MAX - real_size &&
        check_and_reclaim_memory(requested_buffer_size + real_size)) {
      device_malloc((void **)&buffer_dev_1_b, real_size);
      device_malloc((void **)&buffer_dev_2_b, requested_buffer_size);
      allocated_buffer_size_b = requested_buffer_size;
    }
    // A failed optional reservation keeps the serial GPU path, even when
    // host_fallback_enabled is false.
  }
  return 1;
}

/*******************************************************************************
 * \brief Checks size of the compact device scratch buffer.
 ******************************************************************************/
static int pw_gpu_ensure_gbuf_c(const size_t requested_size) {
  assert(is_initialized);
  scratch_epoch++;
  if (requested_size > allocated_gbuf_size) {
    if (!check_and_reclaim_memory(requested_size - allocated_gbuf_size)) {
      return memory_limit_exceeded();
    }
    offloadStreamSynchronize(stream);
    device_free(gbuf_dev, allocated_gbuf_size);
    device_malloc((void **)&gbuf_dev, requested_size);
    allocated_gbuf_size = requested_size;
  }
  return 1;
}

static int pw_gpu_ensure_xc_c(const size_t requested_size) {
  const size_t growth = requested_size > allocated_xc_size
                            ? requested_size - allocated_xc_size : 0;
  protected_xc_buffer = true;
  const int fits = check_and_reclaim_memory(growth);
  protected_xc_buffer = false;
  if (!fits) return memory_limit_exceeded();
  if (growth) {
    release_xc_buffer();
    const size_t lo = xc_lo_bytes(requested_size);
    device_malloc((void **)&xc_dev, lo);
    if (requested_size > lo) {
      device_malloc((void **)&xc_dev_hi, requested_size - lo);
    }
    allocated_xc_size = requested_size;
    xc_epoch++;
  }
  return 1;
}

/*******************************************************************************
 * \brief Residency core: slot bookkeeping, fingerprints and optional memory.
 ******************************************************************************/
#include "pw_gpu_res_ops.h"

static void res_natural_download(const double *src, double *dst,
                                  const int *map, const size_t n) {
  assert(res_natural_dev != NULL && res_natural_capacity != 0);
  int *coords = (int *)(res_natural_dev + 2 * res_natural_capacity);
  for (size_t first = 0; first < n; first += res_natural_capacity) {
    const size_t count = n - first < res_natural_capacity
                             ? n - first : res_natural_capacity;
    offloadMemcpyAsyncHtoD(coords, map + first, count * sizeof(int), stream);
    pw_gpu_launch_res_natural_extract_prefix(src, res_natural_dev, coords,
                                              count, stream);
    offloadMemcpyAsyncDtoH(dst + 2 * first, res_natural_dev,
                           2 * count * sizeof(double), stream);
  }
}

// No allocation is needed to materialize a lazy result, including under memory
// pressure. Only equal-width slots (density and Vtot) can be authoritative.
static void res_sync(const int id) {
  res_slot *s = &res[id];
  if (!s->dev_authoritative) return;
  assert(s->state == RES_VALID && s->dev != NULL && s->host != NULL);
  assert(s->width == s->host_width && (s->owned_bytes != 0 || s->epoch == xc_epoch));
  offload_activate_chosen_device();
  const size_t bytes = s->n * (size_t)s->width * sizeof(double);
  if (s->natural_g) {
    res_natural_download(s->dev, (double *)s->host, res_natural.map, s->n);
  } else {
    offloadMemcpyAsyncDtoH((void *)s->host, s->dev, bytes, stream);
  }
  offloadStreamSynchronize(stream);
  s->dev_authoritative = false;
  res_stats.syncs++;
  res_stats.sync_bytes += bytes;
}

// id == -1 searches all slots, as a real PW can be either RHO_R or VXC.
// A pointer into an array also synchronizes its owner before a slice is used.
void pw_gpu_res_ensure_host_c(const int id, void *host_ptr) {
  if (host_ptr == NULL || id < -1 || id >= RES_NSLOTS) return;
#pragma omp critical(pw_gpu_res_host)
  {
    if (is_initialized) {
      for (int i = 0; i < RES_NSLOTS; i++) {
        res_slot *s = &res[i];
        const uintptr_t p = (uintptr_t)host_ptr, base = (uintptr_t)s->host;
        if ((id == -1 || id == i) && s->dev_authoritative && p >= base &&
            p - base < s->n * (size_t)s->host_width * sizeof(double)) res_sync(i);
      }
    }
  }
}

static void res_invalidate(const int id) {
  res[id].dev_authoritative = false;
  res[id].state = RES_INVALID;
  res[id].vtot = false;
  res[id].natural_g = false;
}

static void res_natural_forget(void) {
  for (int id = 0; id < RES_NSLOTS; id++) {
    if (res[id].natural_g) {
      res_sync(id);
      res_invalidate(id);
    }
  }
  // Coarse-grid reference indices depend on the target's ordering too.
  for (int i = 0; i < PW_GPU_GRID_CACHE_SIZE; i++) {
    if (grid_cache[i].gidx_natural_map != NULL)
      grid_cache_clear_entry(&grid_cache[i]);
  }
  if (res_hartree.natural_g) res_hartree.pending = false;
  memset(&res_natural, 0, sizeof(res_natural));
}

static void res_invalidate_borrowed(void) {
  // Before the XC grids are reused, a live result is copied to its host array
  // unless its consumer has just read it (see res_retire_borrowed).
  for (int id = 0; id <= RES_VXC; id++) {
    res_sync(id);
    res_invalidate(id);
  }
  xc_epoch++;
}

// The fused XC pipeline is the device consumer of a lazy density: inside the
// granted window the host copies of RHO_G/RHO_R are dead once XC has read them
// (the next density overwrites them, the SCF exit is never reached with a
// consumed density). Anything else that is still live is synchronized first.
static void res_retire_borrowed(const unsigned consumed) {
  for (int id = 0; id <= RES_VXC; id++) {
    const bool dead = res_lazy_grant &&
        (id == RES_RHO_G || id == RES_RHO_R) && (consumed & (1u << id));
    if (dead) {
      // Clear authority before invalidating the shared XC storage. These
      // densities have reached their last consumer in the granted window.
      res[id].dev_authoritative = false;
    } else {
      res_sync(id);
    }
    res_invalidate(id);
  }
  xc_epoch++;
}

static void res_free_owned(const int id) {
  res_slot *s = &res[id];
  if (s->owned_bytes != 0) {
    res_sync(id); // a lazy owned result (V_H_R) must survive eviction
    offloadStreamSynchronize(stream);
    offloadStreamSynchronize(stream_copy);
    device_free(s->dev, s->owned_bytes);
    s->dev = NULL;
    s->owned_bytes = 0;
  }
  res_invalidate(id);
}

static void res_free_small(void) {
  if (res_small_dev != NULL) {
    offloadStreamSynchronize(stream);
    offloadStreamSynchronize(stream_copy);
    device_free(res_small_dev, res_small_bytes);
    res_small_dev = NULL;
    res_small_bytes = 0;
  }
}

// Owned slots are the cheapest data to recreate: evict them before any
// scratch that an operation requires. Pinned slots are in use right now.
static bool res_evict_one(void) {
  static const int order[] = {RES_INFL, RES_RHO_CORE, RES_V_H_R};
  for (size_t i = 0; i < sizeof(order) / sizeof(order[0]); i++) {
    const int id = order[i];
    if (res[id].owned_bytes != 0 && (res_pinned & (1u << id)) == 0) {
      res_free_owned(id);
      return true;
    }
  }
  return false;
}

static void res_evict_all(void) {
  for (int id = RES_V_H_R; id < RES_NSLOTS; id++) {
    res_free_owned(id);
  }
  res_free_small();
  res_invalidate_borrowed();
  offloadStreamSynchronize(stream);
  device_free(res_natural_dev, res_natural_capacity * (2 * sizeof(double) + sizeof(int)));
  res_natural_dev = NULL;
  res_natural_capacity = 0;
}

// Free device memory as reported by the runtime (false: unknown).
static bool res_device_free_memory(size_t *free_bytes) {
#if defined(PW_GPU_TEST_FREE_MEMORY)
  *free_bytes = PW_GPU_TEST_FREE_MEMORY();
  return true;
#elif defined(__OFFLOAD_HIP)
  size_t total;
  return hipMemGetInfo(free_bytes, &total) == hipSuccess;
#elif defined(__OFFLOAD_CUDA)
  size_t total;
  return cudaMemGetInfo(free_bytes, &total) == cudaSuccess;
#else
  (void)free_bytes;
  return false;
#endif
}

// Reclaim idle caches/optional work buffers for persistent Hartree inputs, but
// never evict live density, another owned slot, or the active FFT scratch.
// Leave res_reserve_bytes free for the FFT and other libraries.
static bool res_ensure_owned(const int id, const size_t bytes) {
  res_slot *s = &res[id];
  if (s->owned_bytes == bytes && s->dev != NULL) {
    return true;
  }
  res_free_owned(id);
  if (bytes == 0) {
    return false;
  }
  read_memory_environment();
  for (;;) {
    size_t free_bytes;
    const bool budget_ok = max_gpu_memory_bytes == 0
        ? bytes <= SIZE_MAX - current_gpu_allocated_bytes
        : current_gpu_allocated_bytes <= max_gpu_memory_bytes &&
          bytes <= max_gpu_memory_bytes - current_gpu_allocated_bytes;
    const bool device_ok = !res_device_free_memory(&free_bytes) ||
        (free_bytes >= res_reserve_bytes && bytes <= free_bytes - res_reserve_bytes);
    if (budget_ok && device_ok) break;
    if (allocated_buffer_size_b != 0) {
      release_double_buffers();
      continue;
    }
    grid_cache_entry *oldest = NULL;
    for (int i = 0; i < PW_GPU_GRID_CACHE_SIZE; i++) {
      grid_cache_entry *e = &grid_cache[i];
      if (e->grid_id != 0 && e != protected_grid_entry &&
          (oldest == NULL || e->last_use < oldest->last_use)) oldest = e;
    }
    if (oldest == NULL) return false;
    grid_cache_clear_entry(oldest);
  }
  device_malloc((void **)&s->dev, bytes);
  s->owned_bytes = bytes;
  return true;
}

// Small mandatory scratch (fingerprints, DDAPC parameters). NULL on failure.
static double *res_small(const size_t doubles) {
  if (doubles > SIZE_MAX / sizeof(double) - PW_GPU_MIB) {
    memory_limit_exceeded();
    return NULL;
  }
  const size_t bytes = doubles * sizeof(double);
  if (bytes <= res_small_bytes) {
    return res_small_dev;
  }
  res_free_small();
  const size_t rounded = (bytes + PW_GPU_MIB - 1) / PW_GPU_MIB * PW_GPU_MIB;
  if (!check_and_reclaim_memory(rounded)) {
    memory_limit_exceeded();
    return NULL;
  }
  device_malloc((void **)&res_small_dev, rounded);
  res_small_bytes = rounded;
  return res_small_dev;
}

static bool res_bits_equal(const double a, const double b) {
  return memcmp(&a, &b, sizeof(double)) == 0;
}

static bool res_sample_buffer(res_slot *s, const size_t doubles) {
  if (doubles > s->sample_capacity) {
    double *p = realloc(s->samples, doubles * sizeof(double));
    if (p == NULL) {
      return false;
    }
    s->samples = p;
    s->sample_capacity = doubles;
  }
  return true;
}

// The fingerprint of a host array laid out like slot s.
static void res_gather_host(const res_slot *s, const double *h, double *out) {
  const size_t count = pw_res_sample_count(s->n);
  for (size_t k = 0; k < count; k++) {
    const size_t e = pw_res_sample_index(k, s->n);
    for (int c = 0; c < s->width; c++) {
      out[(size_t)s->width * k + c] = h[(size_t)s->host_width * e + c];
    }
  }
}

static bool res_samples_match(const res_slot *s, const double *h) {
  const size_t count = pw_res_sample_count(s->n);
  int mismatch = 0;
#pragma omp parallel for reduction(| : mismatch) if (count > 4096)
  for (size_t k = 0; k < count; k++) {
    const size_t e = pw_res_sample_index(k, s->n);
    for (int c = 0; c < s->width; c++) {
      if (!res_bits_equal(s->samples[(size_t)s->width * k + c],
                          h[(size_t)s->host_width * e + c])) {
        mismatch = 1;
      }
    }
    // A compact real copy of a complex array assumes zero imaginary parts.
    if (s->host_width == 2 && s->width == 1 &&
        !res_bits_equal(h[2 * e + 1], 0.0)) {
      mismatch = 1;
    }
  }
  return mismatch == 0;
}

// CP2K_PW_GPU_RES_VERIFY: compare the complete device copy (diagnostics).
static bool res_verify_full(const res_slot *s, const double *h) {
  const size_t doubles = s->n * (size_t)s->width;
  double *copy = malloc(doubles * sizeof(double) + 1);
  if (copy == NULL) {
    return false;
  }
  offloadMemcpyAsyncDtoH(copy, s->dev, doubles * sizeof(double), stream);
  offloadStreamSynchronize(stream);
  bool ok = true;
  for (size_t e = 0; e < s->n && ok; e++) {
    const size_t j = s->natural_g ? (size_t)res_natural.map[e] : e;
    for (int c = 0; c < s->width; c++) {
      ok = ok && res_bits_equal(copy[(size_t)s->width * j + c],
                                h[(size_t)s->host_width * e + c]);
    }
    if (s->host_width == 2 && s->width == 1) {
      ok = ok && res_bits_equal(h[2 * e + 1], 0.0);
    }
  }
  free(copy);
  if (!ok) {
    fprintf(stderr, "[PW_GPU] Residency verification failed for %s.\n",
            res_names[s - res]);
  }
  return ok;
}

// Bind the storage and layout; registration below establishes its validity.
static void res_bind(const int id, const void *host, const size_t n,
                     const int width, const int host_width, double *dev) {
  res_slot *s = &res[id];
  s->host = host;
  s->n = n;
  s->width = width;
  s->host_width = host_width;
  if (s->owned_bytes == 0) {
    s->dev = dev;
  }
  assert(s->dev == dev);
  s->epoch = xc_epoch;
  s->vtot = false;
  s->natural_g = false;
  s->dev_authoritative = false;
  s->state = RES_INVALID;
}

static void res_register(const int id, const void *host, const size_t n,
                         const int width, const int host_width, double *dev) {
  res_bind(id, host, n, width, host_width, dev);
  res_slot *s = &res[id];
  if (n != 0 && res_sample_buffer(s, pw_res_sample_count(n) * (size_t)width)) {
    res_gather_host(s, host, s->samples);
    s->state = RES_VALID;
  }
}

// Samples and device scratch have already been reserved before output writes.
static void res_register_lazy(const int id, void *host, const size_t n,
                              const int width, double *dev, double *scratch) {
  res_bind(id, host, n, width, width, dev);
  res_slot *s = &res[id];
  pw_gpu_launch_res_sample(dev, scratch, n, width, stream);
  offloadMemcpyAsyncDtoH(s->samples, scratch,
                        pw_res_sample_count(n) * (size_t)width * sizeof(double), stream);
  s->state = RES_VALID;
  s->dev_authoritative = true;
}

// Device copy of a host array, or NULL if it is not (exactly) resident.
static double *res_lookup(const int id, const void *host, const size_t n,
                          const bool vtot) {
  res_slot *s = &res[id];
  if (!opt_res || host == NULL) {
    return NULL;
  }
  bool ok = s->state == RES_VALID && s->host == host && s->n == n &&
            s->dev != NULL && s->vtot == vtot &&
            (s->owned_bytes != 0 || s->epoch == xc_epoch);
  if (ok && !s->dev_authoritative) {
    ok = res_samples_match(s, host);
    ok = ok && (!opt_res_verify || res_verify_full(s, host));
  }
  if (ok) {
    s->hits++;
    res_stats.upload_saved += n * (size_t)s->width * sizeof(double);
    return s->dev;
  }
  s->misses++;
  if (s->host == host) {
    // A layout mismatch can send a live lazy result to a host fallback.
    pw_gpu_res_ensure_host_c(id, (void *)host);
    res_invalidate(id);
  }
  return NULL;
}

static double *res_lookup_g(const int id, const void *host, const size_t n,
                            const bool natural) {
  if (res[id].natural_g != natural) {
    res_sync(id);
    res_invalidate(id);
  }
  return res_lookup(id, host, n, false);
}

// Check the Fortran object's identity before reducing the single resident slot.
int pw_gpu_res_matches_c(const int id, const void *host, const size_t n) {
  if (!is_initialized || id < 0 || id >= RES_NSLOTS || omp_get_num_threads() != 1) return 0;
  return res_lookup(id, host, n, false) != NULL;
}

int pw_gpu_res_integrate_r_c(double *total_rho_out, const double dvol) {
  if (!is_initialized || !opt_res_density || total_rho_out == NULL ||
      !isfinite(dvol) || omp_get_num_threads() != 1) return 0;
  res_slot *s = &res[RES_RHO_R];
  if (s->n == 0 || res_lookup(RES_RHO_R, s->host, s->n, false) == NULL) return 0;
  offload_activate_chosen_device();
  const size_t first = pw_gpu_xc_reduce_size(s->n);
  const size_t second = pw_gpu_xc_reduce_size(first);
  protected_xc_buffer = true;
  double *partial = res_small(first + second);
  protected_xc_buffer = false;
  if (partial == NULL) return 0;
  const double *input = s->dev;
  double *output = partial;
  size_t remaining = s->n;
  do {
    pw_gpu_launch_xc_reduce(input, output, remaining, stream);
    remaining = pw_gpu_xc_reduce_size(remaining);
    input = output;
    output = output == partial ? partial + first : partial;
  } while (remaining > 1);
  double sum;
  offloadMemcpyAsyncDtoH(&sum, input, sizeof(double), stream);
  offloadStreamSynchronize(stream);
  *total_rho_out = sum * dvol;
  return 1;
}

static void res_print_stats(void) {
  if (!opt_res_stats || (res_stats.density + res_stats.hartree +
                         res_stats.potential + res_stats.xc_in +
                         res_stats.density_fail + res_stats.hartree_fail +
                         res_stats.potential_fail) == 0) {
    return;
  }
  fprintf(stderr,
          "[PW_GPU] residency: density %lu (fallback %lu), hartree %lu "
          "(fallback %lu), xc-input %lu, vtot %lu (fallback %lu), potential "
          "%lu (fallback %lu), uploads avoided %.1f GB\n",
          res_stats.density, res_stats.density_fail, res_stats.hartree,
          res_stats.hartree_fail, res_stats.xc_in, res_stats.vtot,
          res_stats.vtot_fail, res_stats.potential, res_stats.potential_fail,
          (double)res_stats.upload_saved / 1e9);
  fprintf(stderr,
          "[PW_GPU] residency lazy: density %lu, vxc %lu, v_h %lu, discarded %lu, "
          "host syncs %lu (%.1f GB), downloads avoided %.1f GB\n",
          res_stats.lazy_density, res_stats.lazy_vxc, res_stats.lazy_vhr,
          res_stats.discards, res_stats.syncs, (double)res_stats.sync_bytes / 1e9,
          (double)res_stats.download_saved / 1e9);
  fprintf(stderr, "[PW_GPU] residency slots (hits/misses):");
  for (int id = 0; id < RES_NSLOTS; id++) {
    fprintf(stderr, " %s %lu/%lu", res_names[id], res[id].hits, res[id].misses);
  }
  fprintf(stderr, "\n");
}

/*******************************************************************************
 * \brief Public switches and invalidation for the Fortran orchestration.
 *        what: 1 density, 2 Hartree, 3 XC input, 4 potential/V_TOT.
 ******************************************************************************/
int pw_gpu_res_enabled_c(const int what) {
  if (!is_initialized || !opt_res) {
    return 0;
  }
  switch (what) {
  case 1:
    return opt_res_density;
  case 2:
    return opt_res_hartree;
  case 3:
    return opt_res_xcin;
  case 4:
    return opt_res_pot;
  default:
    return 0;
  }
}

void pw_gpu_res_invalidate_c(void) {
  if (!is_initialized) {
    return;
  }
  // The explicit SCF/forces boundary preserves live results for host consumers.
  for (int id = 0; id < RES_NSLOTS; id++) {
    res_sync(id);
    res_invalidate(id);
  }
  res_hartree.pending = false;
}

/*******************************************************************************
 * \brief Lazy window of the SCF driver. on != 0 grants lazy results until the
 *        call with on == 0, which synchronizes every live result to its host
 *        array (resident copies stay valid). Outside the window all results
 *        are written to the host as soon as they are computed.
 ******************************************************************************/
void pw_gpu_res_set_lazy_c(const int on) {
  if (!is_initialized) return;
  res_density_token = false;
  if (on) {
    res_lazy_grant = opt_res && (opt_res_lazy_density || opt_res_lazy_pot) && !opt_res_verify;
    return;
  }
  res_lazy_grant = false;
  offload_activate_chosen_device();
  for (int id = 0; id < RES_NSLOTS; id++) {
    res_sync(id);
  }
}

// The next density computation (and only that one) may stay on the device.
void pw_gpu_res_defer_density_c(const int on) {
  res_density_token = is_initialized && res_lazy_grant && on != 0;
}

// The host array is about to be released without being read again: forget a
// device copy (live or not) without downloading it.
void pw_gpu_res_discard_c(const void *host_ptr) {
  if (host_ptr == NULL) return;
#pragma omp critical(pw_gpu_res_host)
  {
    if (is_initialized) {
      const uintptr_t p = (uintptr_t)host_ptr;
      for (int i = 0; i < RES_NSLOTS; i++) {
        res_slot *s = &res[i];
        const uintptr_t base = (uintptr_t)s->host;
        if (s->state != RES_INVALID && s->host != NULL && p >= base &&
            p - base < s->n * (size_t)s->host_width * sizeof(double)) {
          if (s->dev_authoritative) {
            res_stats.discards++;
            res_stats.download_saved += s->n * (size_t)s->width * sizeof(double);
          }
          res_invalidate(i);
        }
      }
    }
  }
}

// Reserve the complete operation before uploading or changing any output.
// All parts except an optional trailing int map are double-aligned. Reusing
// compact scratch keeps these operations in the existing budget and lifecycle.
static int pw_ops_reserve(const size_t *counts, const size_t *widths, int nparts,
                          void **parts, size_t *bytes) {
  if (!is_initialized) return 0;
  size_t total = 0;
  for (int i = 0; i < nparts; i++) {
    if (counts[i] > (SIZE_MAX - total) / widths[i])
      return memory_limit_exceeded();
    bytes[i] = counts[i] * widths[i];
    total += bytes[i];
  }
  offload_activate_chosen_device();
  if (!pw_gpu_ensure_gbuf_c(total)) return 0;
  size_t offset = 0;
  for (int i = 0; i < nparts; i++) {
    parts[i] = (char *)gbuf_dev + offset;
    offset += bytes[i];
  }
  return 1;
}

static int pw_grid_op(const double *x, double *y, size_t nx, size_t ny,
    int cx, int cy, const int *map, int scatter, double alpha, double beta,
    bool add) {
  if (ny == 0) return 1;
  // These attempts may be called from routines used in parallel CPU regions.
  // Only a serial caller owns the shared PW scratch for grid operations.
  if (!is_initialized || omp_get_num_threads() != 1) return 0;
  if ((cx != 1 && cx != 2) || (cy != 1 && cy != 2)) return 0;
  const size_t n = nx < ny ? nx : ny;
  if (nx > SIZE_MAX / (sizeof(double) * (size_t)cx) ||
      ny > SIZE_MAX / (sizeof(double) * (size_t)cy))
    return memory_limit_exceeded();
  const size_t xbytes = nx * sizeof(double) * (size_t)cx;
  const size_t ybytes = ny * sizeof(double) * (size_t)cy;
  if (ybytes > SIZE_MAX - xbytes ||
      (map != NULL && n > (SIZE_MAX - xbytes - ybytes) / sizeof(int)))
    return memory_limit_exceeded();
  if (map != NULL) {
    for (size_t i = 0; i < n; i++)
      if (map[i] < 1 || (size_t)map[i] > (scatter ? ny : nx)) return 0;
  }
  // pw_axpy scales Y before reading X on the CPU. Aliased arrays with beta!=1
  // must retain that ordering, including partially overlapping pointer slices.
  if (add && beta != 1.0 && alpha != 0.0) {
    const uintptr_t xp = (uintptr_t)x, yp = (uintptr_t)y;
    if (xp <= yp ? yp - xp < xbytes : xp - yp < ybytes) return 0;
  }
  // Some Fortran compilers promote a real coefficient to complex before
  // scaling: (a,0)*(xr,xi) then evaluates zero*Inf/NaN cross terms. Retain the
  // compiler's behavior for nonfinite complex scaling rather than assuming
  // that two independent real multiplies have identical exceptional results.
  if (add && cy == 2) {
    if (alpha != 0.0 && alpha != 1.0) {
      if (!isfinite(alpha)) return 0;
      for (size_t i = 0; i < xbytes / sizeof(double); i++)
        if (!isfinite(x[i])) return 0;
    }
    if (beta != 0.0 && beta != 1.0) {
      if (!isfinite(beta)) return 0;
      for (size_t i = 0; i < ybytes / sizeof(double); i++)
        if (!isfinite(y[i])) return 0;
    }
  }
  const size_t counts[] = {nx, ny, map != NULL ? n : 0};
  const size_t widths[] = {sizeof(double) * (size_t)cx,
                           sizeof(double) * (size_t)cy, sizeof(int)};
  void *p[3];
  size_t b[3];
  if (!pw_ops_reserve(counts, widths, 3, p, b)) return 0;
  if (b[0] && (!add || alpha != 0.0))
    offloadMemcpyAsyncHtoD(p[0], x, b[0], stream);
  if (add && beta != 0.0) offloadMemcpyAsyncHtoD(p[1], y, b[1], stream);
  if (b[2]) offloadMemcpyAsyncHtoD(p[2], map, b[2], stream);
  if (map != NULL) {
    if (add) {
      pw_gpu_launch_axpy(NULL, p[1], 0, ny, cx, cy, 0.0, beta, stream);
      pw_gpu_launch_remap_axpy(p[0], p[1], p[2], n, cx, cy, scatter,
                               alpha, 1.0, stream);
    } else {
      offloadMemsetAsync(p[1], 0, b[1], stream);
      pw_gpu_launch_remap_copy(p[0], p[1], p[2], n, cx, cy, scatter, alpha, stream);
    }
  } else if (add) {
    pw_gpu_launch_axpy(p[0], p[1], nx, ny, cx, cy, alpha, beta, stream);
  } else {
    pw_gpu_launch_copy(p[0], p[1], nx, ny, cx, cy, alpha, stream);
  }
  offloadMemcpyAsyncDtoH(y, p[1], b[1], stream);
  offloadStreamSynchronize(stream);
  return 1;
}

int pw_gpu_copy_c(const double *x, double *y, size_t nx, size_t ny, int cx,
                  int cy, double alpha) {
  return pw_grid_op(x, y, nx, ny, cx, cy, NULL, 0, alpha, 0.0, false);
}

int pw_gpu_axpy_c(const double *x, double *y, size_t nx, size_t ny, int cx,
                  int cy, double alpha, double beta) {
  return pw_grid_op(x, y, nx, ny, cx, cy, NULL, 0, alpha, beta, true);
}

int pw_gpu_remap_copy_c(const double *x, double *y, size_t nx, size_t ny, int cx,
    int cy, const int *gidx, int scatter, double alpha) {
  if (nx && ny && gidx == NULL) return 0;
  return pw_grid_op(x, y, nx, ny, cx, cy, gidx, scatter, alpha, 0.0, false);
}

int pw_gpu_remap_axpy_c(const double *x, double *y, size_t nx, size_t ny, int cx,
    int cy, const int *gidx, int scatter, double alpha, double beta) {
  if (nx && ny && gidx == NULL) return 0;
  return pw_grid_op(x, y, nx, ny, cx, cy, gidx, scatter, alpha, beta, true);
}

int modify_hartree_pot_gpu_c(double *v, const double *g, const double *gsq,
    const double *r, const double *radii, const double *cv, size_t n,
    size_t natoms, size_t nradii, double gcut2, double vol) {
  if (n == 0) return 1;
  if (!is_initialized || omp_get_num_threads() != 1) return 0;
  if (n > SIZE_MAX / (6 * sizeof(double))) return memory_limit_exceeded();
  // Match the sorted-G CPU loop's EXIT, including untouched coefficients
  // beyond the first vector above GCUT. G=0 is handled in the Fortran caller.
  size_t active = 0;
  while (active < n && !(gsq[active] > gcut2)) active++;
  n = active;
  if (n == 0) return 1;
  if (nradii && natoms > SIZE_MAX / nradii) return memory_limit_exceeded();
  const size_t counts[] = {n, n, n, natoms, nradii, natoms * nradii};
  const size_t widths[] = {2 * sizeof(double), 3 * sizeof(double), sizeof(double),
                           3 * sizeof(double), sizeof(double), sizeof(double)};
  void *p[6];
  size_t b[6];
  if (!pw_ops_reserve(counts, widths, 6, p, b)) return 0;
  const void *h[] = {v, g, gsq, r, radii, cv};
  for (int i = 0; i < 6; i++)
    if (b[i]) offloadMemcpyAsyncHtoD(p[i], h[i], b[i], stream);
  pw_gpu_launch_modify_hartree(p[0], p[1], p[2], p[3], p[4], p[5], n,
                               natoms, nradii, gcut2, vol, stream);
  offloadMemcpyAsyncDtoH(v, p[0], b[0], stream);
  offloadStreamSynchronize(stream);
  return 1;
}

int pbe_lda_eval_gpu_c(const double *rho, const double *norm, double *e0,
    double *erho, double *enorm, size_t n, int grad, double epsilon,
    double kappa, double beta, double mu, double sc, double sx) {
  if (grad < 0 || grad > 1) return 0;
  if (n == 0) return 1;
  // pbe_lda_calc calls this once from OMP SINGLE, then broadcasts success.
  const size_t counts[] = {n, n, n, grad ? n : 0, grad ? n : 0};
  const size_t widths[] = {sizeof(double), sizeof(double), sizeof(double),
                           sizeof(double), sizeof(double)};
  void *p[5];
  size_t b[5];
  if (!pw_ops_reserve(counts, widths, 5, p, b)) return 0;
  const void *h[] = {rho, norm, e0, erho, enorm};
  for (int i = 0; i < 5; i++)
    if (b[i]) offloadMemcpyAsyncHtoD(p[i], h[i], b[i], stream);
  pw_gpu_launch_pbe_lda(p[0], p[1], p[2], p[3], p[4], n, grad, epsilon,
                        kappa, beta, mu, sc, sx, stream);
  offloadMemcpyAsyncDtoH(e0, p[2], b[2], stream);
  if (grad) {
    offloadMemcpyAsyncDtoH(erho, p[3], b[3], stream);
    offloadMemcpyAsyncDtoH(enorm, p[4], b[4], stream);
  }
  offloadStreamSynchronize(stream);
  return 1;
}

/*******************************************************************************
 * \brief Returns the cache entry of a grid, uploading g_hatmap if necessary.
 *        The upload is enqueued on the pw_gpu stream, i.e. it is ordered
 *        before all subsequent kernels that use the map.
 ******************************************************************************/
static grid_cache_entry *grid_cache_lookup_or_upload(
    const int grid_id, const int *map_host, const int ngpts, const int nmaps) {
  assert(is_initialized);
  assert(grid_id > 0);
  grid_cache_entry *hit = NULL;
  for (int i = 0; i < PW_GPU_GRID_CACHE_SIZE; i++) {
    grid_cache_entry *e = &grid_cache[i];
    if (e->grid_id != grid_id) {
      continue;
    }
    if (e->map_host == map_host && e->ngpts == ngpts && e->nmaps == nmaps) {
      hit = e;
    } else {
      grid_cache_clear_entry(e); // same id, different layout: never reuse
    }
  }
  if (hit == NULL) {
    // Pick an empty slot, otherwise evict the least recently used one.
    hit = &grid_cache[0];
    for (int i = 0; i < PW_GPU_GRID_CACHE_SIZE; i++) {
      grid_cache_entry *e = &grid_cache[i];
      if (e->grid_id == 0) {
        hit = e;
        break;
      }
      if (e->last_use < hit->last_use) {
        hit = e;
      }
    }
    grid_cache_clear_entry(hit);
    const size_t map_size = sizeof(int) * (size_t)nmaps * (size_t)ngpts;
    if (!check_and_reclaim_memory(map_size)) {
      memory_limit_exceeded();
      return NULL;
    }
    device_malloc((void **)&hit->map_dev, map_size);
    offloadMemcpyAsyncHtoD(hit->map_dev, map_host, map_size, stream);
    hit->grid_id = grid_id;
    hit->ngpts = ngpts;
    hit->nmaps = nmaps;
    hit->map_host = map_host;
  }
  hit->last_use = ++grid_cache_clock;
  return hit;
}

/*******************************************************************************
 * \brief Returns the device copy of the g-vectors g(3,ngpts) of a grid.
 ******************************************************************************/
static const double *grid_cache_get_g(grid_cache_entry *e,
                                      const double *g_host) {
  if (e == NULL) {
    return NULL;
  }
  if (e->g_dev != NULL && e->g_host == g_host) {
    return e->g_dev;
  }
  if (e->g_dev != NULL) {
    offloadStreamSynchronize(stream);
    device_free(e->g_dev, 3 * sizeof(double) * (size_t)e->ngpts);
    e->g_dev = NULL;
  }
  const size_t g_size = 3 * sizeof(double) * (size_t)e->ngpts;
  protected_grid_entry = e;
  const int fits = check_and_reclaim_memory(g_size);
  protected_grid_entry = NULL;
  if (!fits) {
    memory_limit_exceeded();
    return NULL;
  }
  device_malloc((void **)&e->g_dev, g_size);
  offloadMemcpyAsyncHtoD(e->g_dev, g_host, g_size, stream);
  e->g_host = g_host;
  return e->g_dev;
}

/*******************************************************************************
 * \brief Returns the device copy of the reference-grid index gidx(ngpts) of a
 *        grid. All entries are checked once to lie in [1, bound].
 ******************************************************************************/
static const int *grid_cache_get_gidx(grid_cache_entry *e, const int *gidx_host,
                                      const int bound, const int *natural_map) {
  if (e == NULL || gidx_host == NULL) {
    return NULL;
  }
  if (e->ngpts <= 0) {
    return NULL;
  }
  if (e->gidx_dev != NULL && e->gidx_host == gidx_host && e->gidx_bound == bound &&
      e->gidx_natural_map == natural_map) {
    return e->gidx_dev;
  }
  if (e->gidx_dev != NULL) {
    offloadStreamSynchronize(stream);
    device_free(e->gidx_dev, sizeof(int) * (size_t)e->ngpts);
    e->gidx_dev = NULL;
  }
  for (int i = 0; i < e->ngpts; i++) {
    if (gidx_host[i] < 1 || gidx_host[i] > bound) {
      return NULL;
    }
  }
  const size_t size = sizeof(int) * (size_t)e->ngpts;
  int *translated = NULL;
  if (natural_map != NULL) {
    translated = malloc(size);
    if (translated == NULL) return NULL;
    for (int i = 0; i < e->ngpts; i++)
      translated[i] = natural_map[gidx_host[i] - 1] + 1;
  }
  protected_grid_entry = e;
  const int fits = check_and_reclaim_memory(size);
  protected_grid_entry = NULL;
  if (!fits) {
    free(translated);
    memory_limit_exceeded();
    return NULL;
  }
  device_malloc((void **)&e->gidx_dev, size);
  offloadMemcpyAsyncHtoD(e->gidx_dev, translated != NULL ? translated : gidx_host,
                         size, stream);
  if (translated != NULL) {
    offloadStreamSynchronize(stream);
    free(translated);
  }
  e->gidx_host = gidx_host;
  e->gidx_bound = bound;
  e->gidx_natural_map = natural_map;
  return e->gidx_dev;
}

/*******************************************************************************
 * \brief Returns a device pointer to g_hatmap. Uses the grid cache if enabled,
 *        otherwise uploads legacy_map_size bytes into the shared map buffer.
 ******************************************************************************/
static const int *get_map_dev(const int grid_id, const int *map_host,
                              const int ngpts, const int nmaps,
                              const size_t legacy_map_size) {
  if (opt_map_cache && grid_id > 0 && nmaps >= 1) {
    grid_cache_entry *e = grid_cache_lookup_or_upload(grid_id, map_host, ngpts, nmaps);
    return e == NULL ? NULL : e->map_dev;
  }
  // Legacy behaviour: upload the map into the shared buffer for every call.
  if (!pw_gpu_ensure_buffer_c(0, legacy_map_size, false)) {
    return NULL;
  }
  offloadMemcpyAsyncHtoD(ghatmap_dev, map_host, legacy_map_size, stream);
  return ghatmap_dev;
}

/*******************************************************************************
 * \brief Fetches an fft plan from the cache. Returns NULL if not found.
 * \author Ole Schuett
 ******************************************************************************/
static offload_fftHandle *lookup_plan_from_cache(const int key[4]) {
  assert(is_initialized);
  for (int i = 0; i < PW_GPU_CACHE_SIZE; i++) {
    const int *x = cache[i].key;
    if (x[0] == key[0] && x[1] == key[1] && x[2] == key[2] && x[3] == key[3]) {
      return cache[i].plan;
    }
  }
  return NULL;
}

/*******************************************************************************
 * \brief Adds an fft plan to the cache. Assumes ownership of plan's memory.
 * \author Ole Schuett
 ******************************************************************************/
static void add_plan_to_cache(const int key[4], offload_fftHandle *plan) {
  const int i = cache_oldest_entry;
  cache_oldest_entry = (cache_oldest_entry + 1) % PW_GPU_CACHE_SIZE;
  if (cache[i].plan != NULL) {
    offload_fftDestroy(*cache[i].plan);
    free(cache[i].plan);
  }
  cache[i].key[0] = key[0];
  cache[i].key[1] = key[1];
  cache[i].key[2] = key[2];
  cache[i].key[3] = key[3];
  cache[i].plan = plan;
}

/*******************************************************************************
 * \brief   Performs a scaled double precision complex 1D-FFT many times on
 *          the GPU.
 *          Input/output are DEVICE pointers (data_in, date_out).
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
static void fft_1d(const int direction, const int n, const int m,
                   const double *data_in, double *data_out) {
  const int key[4] = {1, direction, n, m}; // first key entry is dimensions
  offload_fftHandle *plan = lookup_plan_from_cache(key);

  if (plan == NULL) {
    int nsize[1] = {n};
    int inembed[1] = {n}; // Valid FFT embedding length
    int onembed[1] = {n}; // Valid FFT embedding length
    int batch = m;
    int istride, idist, ostride, odist;
    if (direction == OFFLOAD_FFT_FORWARD) {
      istride = m;
      idist = 1;
      ostride = 1;
      odist = n;
    } else {
      istride = 1;
      idist = n;
      ostride = m;
      odist = 1;
    }
    plan = malloc(sizeof(cache_entry));
    offload_fftPlanMany(plan, 1, nsize, inembed, istride, idist, onembed,
                        ostride, odist, OFFLOAD_FFT_Z2Z, batch);
    offload_fftSetStream(*plan, stream);
    add_plan_to_cache(key, plan);
  }

  offload_fftExecZ2Z(*plan, data_in, data_out, direction);
}

/*******************************************************************************
 * \brief   Performs a scaled double precision complex 3D-FFT on the GPU.
 *          Input/output is a DEVICE pointer (data).
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
static void fft_3d(const int direction, const int nx, const int ny,
                   const int nz, double *data) {
  const int key[4] = {3, nx, ny, nz}; // first key entry is dimensions
  offload_fftHandle *plan = lookup_plan_from_cache(key);

  if (plan == NULL) {
    plan = malloc(sizeof(cache_entry));
    offload_fftPlan3d(plan, nx, ny, nz, OFFLOAD_FFT_Z2Z);
    offload_fftSetStream(*plan, stream);
    add_plan_to_cache(key, plan);
  }

  offload_fftExecZ2Z(*plan, data, data, direction);
}

int pw_gpu_xc_full_pipeline_configured_c(const double *zin_rho_g,
    const double *din_rho_r, double *dout_vxc_r, double *exc_out,
    const int *ghatmap, const double *g, const int *npts, int ngpts, int nmaps,
    double scale, int grid_id, double epsilon_rho, double kappa, double beta,
    double mu, double scale_ec, double scale_ex, double gradient_cutoff,
    double smooth_range, int spherical) {
  if (!is_initialized || !opt_fused_xc || omp_get_num_threads() != 1) return 0;
  if (npts == NULL || exc_out == NULL || ngpts < 0 ||
      (nmaps != 1 && nmaps != 2) || grid_id <= 0 ||
      !isfinite(gradient_cutoff) || gradient_cutoff < 0.0) return 0;
  size_t nr = 1;
  for (int d = 0; d < 3; d++) {
    if (npts[d] < 0) return 0;
    // Existing FFT conversion/map kernels use int for doubled indices.
    if (npts[d] != 0 && nr > (size_t)(INT_MAX / 2) / (size_t)npts[d])
      return memory_limit_exceeded();
    nr *= (size_t)npts[d];
  }
  if ((size_t)ngpts > nr) return 0;
  if (nr == 0) {
    *exc_out = 0.0;
    return 1;
  }
  if (nr > SIZE_MAX / (8 * sizeof(double))) return memory_limit_exceeded();
  if (din_rho_r == NULL || dout_vxc_r == NULL ||
      (ngpts && (zin_rho_g == NULL || ghatmap == NULL || g == NULL))) return 0;
  const int nrpts = (int)nr;
  const size_t real_size = sizeof(double) * nr;
  const size_t buffer_size = 2 * real_size;
  const size_t gbuf_size = 2 * sizeof(double) * (size_t)ngpts;
  offload_activate_chosen_device();

  // Reserve every required byte before consuming inputs or changing outputs.
  // Optional dual-stream scratch and inactive grids can be reclaimed. Pin the
  // active map while reserving the four real XC grids.
  if (!pw_gpu_ensure_buffer_c(buffer_size, 0, false) ||
      !pw_gpu_ensure_gbuf_c(gbuf_size)) return 0;
  grid_cache_entry *entry = NULL;
  const double *g_dev = NULL;
  if (ngpts) {
    if (!opt_map_cache) pw_gpu_grid_invalidate(grid_id);
    entry = grid_cache_lookup_or_upload(grid_id, ghatmap, ngpts, nmaps);
    if (entry == NULL || (g_dev = grid_cache_get_g(entry, g)) == NULL) return 0;
  }
  protected_grid_entry = entry;
  const int fits = pw_gpu_ensure_xc_c(4 * real_size);
  // A lazy Vxc stays in the first XC grid for its device consumers (V_TOT,
  // potential_pw2rs); only its fingerprint and the energy are downloaded.
  bool lazy_vxc = fits && res_lazy_grant && opt_res_lazy_pot && opt_res_pot;
  double *vxc_samples = NULL;
  if (lazy_vxc) {
    protected_xc_buffer = true;
    vxc_samples = res_small(pw_res_sample_count(nr));
    protected_xc_buffer = false;
    lazy_vxc = vxc_samples != NULL &&
               res_sample_buffer(&res[RES_VXC], pw_res_sample_count(nr));
  }
  protected_grid_entry = NULL;
  if (!fits) return 0;
  const int *map_dev = entry == NULL ? NULL : entry->map_dev;
  double *grad_dev[3] = {xc_dev, xc_dev + nr, xc_dev_hi};
  double *vrho_dev = xc_dev_hi + nr;
  double *rho_dev = buffer_dev_1, *norm_dev = buffer_dev_1 + nr;
  double *e0_dev = buffer_dev_2, *vnorm_dev = buffer_dev_2 + nr;

  // Densities left in the XC grids by pw_gpu_res_density_c replace the
  // uploads. Uploads are issued first, then the device copies; the XC grids
  // are overwritten afterwards, which ends the residency of both densities.
  const double *res_g = (ngpts && opt_res_xcin)
      ? res_lookup(RES_RHO_G, zin_rho_g, (size_t)ngpts, false) : NULL;
  if (res_g != NULL && res[RES_RHO_G].natural_g &&
      (res_natural.grid_id != grid_id ||
       memcmp(res_natural.npts, npts, sizeof(res_natural.npts)))) {
    res_sync(RES_RHO_G);
    res_g = NULL;
  }
  const double *res_r = opt_res_xcin ? res_lookup(RES_RHO_R, din_rho_r, nr, false) : NULL;
  if (ngpts && res_g == NULL) {
    pw_gpu_res_ensure_host_c(RES_RHO_G, (void *)zin_rho_g);
    offloadMemcpyAsyncHtoD(gbuf_dev, zin_rho_g, gbuf_size, stream);
  }
  if (res_r == NULL) {
    pw_gpu_res_ensure_host_c(RES_RHO_R, (void *)din_rho_r);
    offloadMemcpyAsyncHtoD(rho_dev, din_rho_r, real_size, stream);
  }
  if (res_g != NULL) {
    if (res[RES_RHO_G].natural_g) {
      pw_gpu_launch_gather(gbuf_dev, res_g, 1.0, ngpts, map_dev, stream);
    } else {
      pw_gpu_launch_copy(res_g, gbuf_dev, (size_t)ngpts, (size_t)ngpts, 2, 2, 1.0, stream);
    }
  }
  if (res_r != NULL) pw_gpu_launch_copy(res_r, rho_dev, nr, nr, 1, 1, 1.0, stream);
  if (res_g != NULL || res_r != NULL) res_stats.xc_in++;
  res_retire_borrowed((res_g != NULL ? 1u << RES_RHO_G : 0u) |
                      (res_r != NULL ? 1u << RES_RHO_R : 0u));
  for (int d = 0; d < 3; d++) {
    offloadMemsetAsync(buffer_dev_2, 0, buffer_size, stream);
    if (ngpts)
      pw_gpu_launch_scatter_deriv(buffer_dev_2, gbuf_dev, g_dev, d, 1.0,
                                  ngpts, nmaps, map_dev, stream);
    fft_3d(OFFLOAD_FFT_INVERSE, npts[2], npts[1], npts[0], buffer_dev_2);
    pw_gpu_launch_complex_to_real(buffer_dev_2, grad_dev[d], nrpts, stream);
  }
  pw_gpu_launch_norm3(grad_dev[0], grad_dev[1], grad_dev[2], norm_dev, nr, stream);
  // The PBE elemental routine accumulates and skips sub-cutoff densities.
  offloadMemsetAsync(buffer_dev_2, 0, buffer_size, stream);
  offloadMemsetAsync(vrho_dev, 0, real_size, stream);
  pw_gpu_launch_pbe_lda(rho_dev, norm_dev, e0_dev, vrho_dev, vnorm_dev, nr, 1,
                        epsilon_rho, kappa, beta, mu, scale_ec, scale_ex, stream);
  pw_gpu_launch_combine_div_input(grad_dev[0], grad_dev[1], grad_dev[2],
                                  vnorm_dev, norm_dev, nr, gradient_cutoff, stream);
  pw_gpu_launch_xc_cutoff(e0_dev, rho_dev, nr, epsilon_rho, smooth_range, stream);

  // Norm and e0 are now dead. Ping-pong compensated reductions in their space
  // until only one scalar remains, kept in norm_dev across the divergence FFTs.
  size_t remaining = nr;
  double *sum_in = e0_dev, *sum_out = norm_dev;
  do {
    pw_gpu_launch_xc_reduce(sum_in, sum_out, remaining, stream);
    remaining = pw_gpu_xc_reduce_size(remaining);
    double *swap = sum_in;
    sum_in = sum_out;
    sum_out = swap;
  } while (remaining > 1);
  if (sum_in != norm_dev)
    pw_gpu_launch_copy(sum_in, norm_dev, 1, 1, 1, 1, 1.0, stream);

  if (ngpts) offloadMemsetAsync(gbuf_dev, 0, gbuf_size, stream);
  for (int d = 0; d < 3; d++) {
    pw_gpu_launch_real_to_complex(grad_dev[d], buffer_dev_2, nrpts, stream);
    fft_3d(OFFLOAD_FFT_FORWARD, npts[2], npts[1], npts[0], buffer_dev_2);
    if (ngpts)
      pw_gpu_launch_gather_deriv_acc(gbuf_dev, buffer_dev_2, g_dev, d, scale,
                                     ngpts, map_dev, stream);
  }
  offloadMemsetAsync(buffer_dev_2, 0, buffer_size, stream);
  if (ngpts)
    pw_gpu_launch_scatter(buffer_dev_2, gbuf_dev, 1.0, ngpts, nmaps, map_dev, stream);
  fft_3d(OFFLOAD_FFT_INVERSE, npts[2], npts[1], npts[0], buffer_dev_2);
  // All three gradient inputs have been consumed; reuse the first for Vxc.
  double *dout_dev = grad_dev[0];
  pw_gpu_launch_complex_to_real(buffer_dev_2, dout_dev, nrpts, stream);
  pw_gpu_launch_add_vrho(dout_dev, vrho_dev, nr, stream);

  // xc_vxc_pw_create filters the entire potential, including dE/drho.
  if (spherical && ngpts) {
    pw_gpu_launch_real_to_complex(dout_dev, buffer_dev_2, nrpts, stream);
    fft_3d(OFFLOAD_FFT_FORWARD, npts[2], npts[1], npts[0], buffer_dev_2);
    pw_gpu_launch_gather(gbuf_dev, buffer_dev_2, scale, ngpts, map_dev, stream);
    offloadMemsetAsync(buffer_dev_2, 0, buffer_size, stream);
    pw_gpu_launch_scatter(buffer_dev_2, gbuf_dev, 1.0, ngpts, nmaps, map_dev, stream);
    fft_3d(OFFLOAD_FFT_INVERSE, npts[2], npts[1], npts[0], buffer_dev_2);
    pw_gpu_launch_complex_to_real(buffer_dev_2, dout_dev, nrpts, stream);
  } else if (spherical) {
    // An empty reciprocal-space filter removes the local PBE potential too.
    offloadMemsetAsync(dout_dev, 0, real_size, stream);
  }
  pw_gpu_launch_xc_cutoff(dout_dev, rho_dev, nr, epsilon_rho * smooth_range,
                          smooth_range, stream);
  // These are the only downloads: the final potential and one energy scalar.
  offloadMemcpyAsyncDtoH(exc_out, norm_dev, sizeof(double), stream);
  if (lazy_vxc) {
    res_register_lazy(RES_VXC, dout_vxc_r, nr, 1, dout_dev, vxc_samples);
    offloadStreamSynchronize(stream);
    res_stats.lazy_vxc++;
    res_stats.download_saved += real_size;
    if (!opt_map_cache) pw_gpu_grid_invalidate(grid_id);
    return 1;
  }
  offloadMemcpyAsyncDtoH(dout_vxc_r, dout_dev, real_size, stream);
  offloadStreamSynchronize(stream);
  if (!opt_map_cache) pw_gpu_grid_invalidate(grid_id);
  // Vxc stays in the first XC grid for pw_gpu_res_vtot_prepare_c.
  if (opt_res_pot) res_register(RES_VXC, dout_vxc_r, nr, 1, 1, dout_dev);
  return 1;
}

int pw_gpu_xc_full_pipeline_c(const double *zin_rho_g, const double *din_rho_r,
    double *dout_vxc_r, double *exc_out, const int *ghatmap, const double *g,
    const int *npts, int ngpts, int nmaps, double scale, int grid_id,
    double epsilon_rho, double kappa, double beta, double mu,
    double scale_ec, double scale_ex) {
  return pw_gpu_xc_full_pipeline_configured_c(zin_rho_g, din_rho_r, dout_vxc_r,
      exc_out, ghatmap, g, npts, ngpts, nmaps, scale, grid_id, epsilon_rho,
      kappa, beta, mu, scale_ec, scale_ex, epsilon_rho, 0.0, 0);
}

/*******************************************************************************
 * \brief Number of points of an FFT grid; false if it is empty or too large
 *        for the int-indexed conversion kernels.
 ******************************************************************************/
static bool res_grid_points(const int *npts, size_t *nr) {
  size_t n = 1;
  for (int d = 0; d < 3; d++) {
    if (npts[d] <= 0 || n > (size_t)(INT_MAX / 2) / (size_t)npts[d]) {
      return false;
    }
    n *= (size_t)npts[d];
  }
  *nr = n;
  return true;
}

#define PW_GPU_RES_MAX_LEVELS 16

static bool res_imag_zero(const double *h, const size_t n) {
  int bad = 0;
#pragma omp parallel for reduction(| : bad)
  for (size_t i = 0; i < n; i++) {
    uint64_t bits;
    memcpy(&bits, &h[2 * i + 1], sizeof(bits));
    bad |= bits != 0;
  }
  return bad == 0;
}

static void res_host_copy(double *dst, const double *src, const size_t n) {
  if (dst == src || n == 0) {
    return;
  }
  const size_t block = 1 << 18;
  const size_t nblocks = (n + block - 1) / block;
#pragma omp parallel for schedule(static)
  for (size_t b = 0; b < nblocks; b++) {
    const size_t lo = b * block, len = n - lo < block ? n - lo : block;
    memcpy(dst + lo, src + lo, len * sizeof(double));
  }
}

static bool res_natural_prepare(const int grid_id, const int *map,
                                 const int *npts, const size_t n) {
  if (res_natural.grid_id != grid_id || res_natural.map != map ||
      res_natural.n != n || memcmp(res_natural.npts, npts, sizeof(res_natural.npts))) {
    // Full-box residency requires a permutation, not a cutoff or half-space
    // map. Validate once per immutable grid, without allocating a device map.
    unsigned char *seen = calloc((n + 7) / 8, 1);
    if (seen == NULL) return false;
    bool valid = true;
    for (size_t i = 0; i < n; i++) {
      if (map[i] < 0 || (size_t)map[i] >= n ||
          (seen[(size_t)map[i] / 8] & (1u << (map[i] % 8)))) {
        valid = false;
        break;
      }
      seen[(size_t)map[i] / 8] |= (unsigned char)(1u << (map[i] % 8));
    }
    free(seen);
    if (!valid) return false;
    res_natural_forget();
    res_natural.grid_id = grid_id;
    res_natural.map = map;
    res_natural.n = n;
    memcpy(res_natural.npts, npts, sizeof(res_natural.npts));
  }
  // Drop an old sorted-G map and its derivative vectors before reserving
  // scratch or optional RHO_CORE/INFL. The coarse-grid maps remain useful.
  for (int i = 0; i < PW_GPU_GRID_CACHE_SIZE; i++) {
    if (grid_cache[i].grid_id == grid_id) grid_cache_clear_entry(&grid_cache[i]);
  }
  const size_t count = n < PW_GPU_RES_NATURAL_CHUNK ? n : PW_GPU_RES_NATURAL_CHUNK;
  if (count > res_natural_capacity) {
    const size_t unit = 2 * sizeof(double) + sizeof(int);
    if (!check_and_reclaim_memory((count - res_natural_capacity) * unit)) return false;
    offloadStreamSynchronize(stream);
    device_free(res_natural_dev, res_natural_capacity * unit);
    device_malloc((void **)&res_natural_dev, count * unit);
    res_natural_capacity = count;
  }
  return true;
}

// The host remains in sorted G order. Convert immutable inputs only on a
// cache miss; the streaming Hartree kernels never read a full device map.
static bool res_natural_upload(double *dev, const double *host, const int width) {
  double *box = malloc(res_natural.n * (size_t)width * sizeof(double));
  if (box == NULL) return false;
#pragma omp parallel for schedule(static) if (res_natural.n >= 4096)
  for (size_t i = 0; i < res_natural.n; i++) {
    const size_t j = (size_t)res_natural.map[i];
    for (int c = 0; c < width; c++) box[(size_t)width * j + c] = host[2 * i + c];
  }
  offloadMemcpyAsyncHtoD(dev, box, res_natural.n * (size_t)width * sizeof(double), stream);
  offloadStreamSynchronize(stream); // box must outlive its asynchronous upload
  free(box);
  return true;
}

static void res_register_natural_lazy(double *host, const size_t n, double *dev) {
  res_bind(RES_RHO_G, host, n, 2, 2, dev);
  res_slot *s = &res[RES_RHO_G];
  s->natural_g = true;
  const size_t count = pw_res_sample_count(n);
  int coords[PW_GPU_RES_PREFIX_SAMPLES + PW_GPU_RES_STRIDED_SAMPLES];
  for (size_t k = 0; k < count; k++)
    coords[k] = res_natural.map[pw_res_sample_index(k, n)];
  int *dev_coords = (int *)(res_natural_dev + 2 * res_natural_capacity);
  offloadMemcpyAsyncHtoD(dev_coords, coords, count * sizeof(int), stream);
  pw_gpu_launch_res_natural_extract_prefix(dev, res_natural_dev, dev_coords, count, stream);
  offloadMemcpyAsyncDtoH(s->samples, res_natural_dev, 2 * count * sizeof(double), stream);
  offloadStreamSynchronize(stream);
  s->state = RES_VALID;
  s->dev_authoritative = true;
}

/*******************************************************************************
 * \brief density_rs2pw (pw_interp) on the device:
 *          rho_g = 0; for each level l: rho_g += P_l gather(scale_l FFT[r_l]);
 *          rho_r = Re FFT^-1[scatter(rho_g)]
 *        P_l is the identity (gidx[l] == NULL, same grid as rho_g) or the
 *        scatter through the one-based reference index gidx[l]. The sequence
 *        of rounded operations is that of the original pw_transfer/pw_axpy
 *        loop. Lazy mode downloads only fingerprints; the results stay in
 *        the XC grids until a host consumer or scratch reuse requires them.
 *        Returns 0 with all host outputs untouched if anything is unsupported.
 ******************************************************************************/
static int res_density(const bool lazy_allowed, const int nlevels,
                       void *const *r_host, void *const *r_dev, const int *npts,
                       const int *ngpts, const int *nmaps, const int *grid_id,
                       void *const *maps, void *const *gidx, const double *scale,
                       const int *t_npts, const int t_ngpts, const int t_nmaps,
                       const int t_grid_id, const int *t_map, double *rho_g_host,
                       double *rho_r_host) {
  if (!is_initialized || !opt_res_density || !opt_map_cache ||
      omp_get_num_threads() != 1) {
    return 0;
  }
  if (nlevels < 1 || nlevels > PW_GPU_RES_MAX_LEVELS || rho_g_host == NULL ||
      rho_r_host == NULL || t_map == NULL || t_ngpts <= 0 ||
      (t_nmaps != 1 && t_nmaps != 2) || t_grid_id <= 0) {
    return 0;
  }
  size_t t_nr, nr[PW_GPU_RES_MAX_LEVELS], roff[PW_GPU_RES_MAX_LEVELS];
  if (!res_grid_points(t_npts, &t_nr) || (size_t)t_ngpts > t_nr) {
    return 0;
  }
  size_t nr_max = t_nr, ng_max = (size_t)t_ngpts, total = 0;
  for (int l = 0; l < nlevels; l++) {
    if (!res_grid_points(npts + 3 * l, &nr[l]) || ngpts[l] <= 0 ||
        (size_t)ngpts[l] > nr[l] || (nmaps[l] != 1 && nmaps[l] != 2) ||
        grid_id[l] <= 0 || (r_host[l] == NULL && (r_dev == NULL || r_dev[l] == NULL)) || maps[l] == NULL) {
      return 0;
    }
    if (gidx[l] == NULL) {
      if (grid_id[l] != t_grid_id || ngpts[l] != t_ngpts) return 0;
    } else if (ngpts[l] >= t_ngpts) {
      return 0;
    }
    roff[l] = total;
    if (r_dev == NULL || r_dev[l] == NULL) total += nr[l];
    nr_max = nr[l] > nr_max ? nr[l] : nr_max;
    ng_max = (size_t)ngpts[l] > ng_max ? (size_t)ngpts[l] : ng_max;
  }
  if (total > 2 * nr_max) {
    return 0; // all real inputs must fit into one complex buffer
  }
  offload_activate_chosen_device();
  // The Fortran entry admits only local grids (one MPI rank owns the box).
  // Sparse and conjugate half-space maps retain their sorted representation.
  const bool natural = opt_res_natural_g && t_nmaps == 1 && (size_t)t_ngpts == t_nr;
  if (natural) {
    for (int l = 0; l < nlevels; l++) {
      if (gidx[l] == NULL && (maps[l] != t_map || nmaps[l] != 1 ||
          memcmp(npts + 3 * l, t_npts, 3 * sizeof(int)))) return 0;
    }
    if (!res_natural_prepare(t_grid_id, t_map, t_npts, t_nr)) return 0;
  }
  if (!pw_gpu_ensure_buffer_c(2 * sizeof(double) * nr_max, 0, false) ||
      !pw_gpu_ensure_gbuf_c(2 * sizeof(double) * ng_max) ||
      !pw_gpu_ensure_xc_c(4 * sizeof(double) * t_nr)) {
    res_stats.density_fail++;
    return 0;
  }
  double *rho_g_dev = xc_dev, *rho_r_dev = xc_dev_hi;
  res_invalidate_borrowed(); // the XC grids are overwritten below
  protected_xc_buffer = true; // grid cache uploads must not release them
  const bool lazy = lazy_allowed && res_lazy_grant && opt_res_lazy_density;
  double *samples = NULL;
  if (lazy) {
    const size_t gs = 2 * pw_res_sample_count((size_t)t_ngpts);
    const size_t rs = pw_res_sample_count(t_nr);
    samples = res_small(gs > rs ? gs : rs);
    if (samples == NULL || !res_sample_buffer(&res[RES_RHO_G], gs) ||
        !res_sample_buffer(&res[RES_RHO_R], rs)) {
      res_stats.density_fail++;
      return 0;
    }
  }

  for (int l = 0; l < nlevels; l++) {
    if (r_dev != NULL && r_dev[l] != NULL) continue;
    offloadMemcpyAsyncHtoD(buffer_dev_1 + roff[l], r_host[l],
                           sizeof(double) * nr[l], stream);
  }
  offloadMemsetAsync(rho_g_dev, 0, 2 * sizeof(double) * (size_t)t_ngpts, stream);
  for (int l = 0; l < nlevels; l++) {
    const double *input = buffer_dev_1 + roff[l];
    if (r_dev != NULL && r_dev[l] != NULL) {
      const grid_buffer_descriptor source = {r_dev[l], sizeof(double) * nr[l]};
      grid_buffer_descriptor view;
      if (!grid_pw_bridge(&source, source.nbytes, &view)) return 0;
      input = view.dev_ptr;
      res_stats.upload_saved += source.nbytes;
    }
    if (natural && gidx[l] == NULL) {
      pw_gpu_launch_real_to_complex(input, buffer_dev_2,
                                    (int)nr[l], stream);
      fft_3d(OFFLOAD_FFT_FORWARD, npts[3 * l + 2], npts[3 * l + 1], npts[3 * l],
             buffer_dev_2);
      pw_gpu_launch_copy(buffer_dev_2, buffer_dev_2, t_nr, t_nr, 2, 2, scale[l], stream);
      pw_gpu_launch_axpy(buffer_dev_2, rho_g_dev, t_nr, t_nr, 2, 2, 1.0, 1.0, stream);
      continue;
    }
    grid_cache_entry *e =
        grid_cache_lookup_or_upload(grid_id[l], maps[l], ngpts[l], nmaps[l]);
    const int *gidx_dev = NULL;
    if (e != NULL && gidx[l] != NULL) {
      gidx_dev = grid_cache_get_gidx(e, gidx[l], t_ngpts, natural ? t_map : NULL);
    }
    if (e == NULL || (gidx[l] != NULL && gidx_dev == NULL)) {
      offloadStreamSynchronize(stream);
      res_stats.density_fail++;
      return 0;
    }
    pw_gpu_launch_real_to_complex(input, buffer_dev_2,
                                  (int)nr[l], stream);
    fft_3d(OFFLOAD_FFT_FORWARD, npts[3 * l + 2], npts[3 * l + 1], npts[3 * l],
           buffer_dev_2);
    pw_gpu_launch_gather(gbuf_dev, buffer_dev_2, scale[l], ngpts[l], e->map_dev,
                         stream);
    if (gidx_dev == NULL) {
      pw_gpu_launch_axpy(gbuf_dev, rho_g_dev, (size_t)ngpts[l], (size_t)ngpts[l],
                         2, 2, 1.0, 1.0, stream);
    } else {
      pw_gpu_launch_remap_axpy(gbuf_dev, rho_g_dev, gidx_dev, (size_t)ngpts[l],
                               2, 2, 1, 1.0, 1.0, stream);
    }
  }
  if (natural) {
    pw_gpu_launch_copy(rho_g_dev, buffer_dev_2, t_nr, t_nr, 2, 2, 1.0, stream);
  } else {
    grid_cache_entry *t =
        grid_cache_lookup_or_upload(t_grid_id, t_map, t_ngpts, t_nmaps);
    if (t == NULL) {
      offloadStreamSynchronize(stream);
      res_stats.density_fail++;
      return 0;
    }
    offloadMemsetAsync(buffer_dev_2, 0, 2 * sizeof(double) * t_nr, stream);
    pw_gpu_launch_scatter(buffer_dev_2, rho_g_dev, 1.0, t_ngpts, t_nmaps,
                          t->map_dev, stream);
  }
  offloadEventRecord(event_compute_done[0], stream); // rho_g complete and read
  fft_3d(OFFLOAD_FFT_INVERSE, t_npts[2], t_npts[1], t_npts[0], buffer_dev_2);
  pw_gpu_launch_complex_to_real(buffer_dev_2, rho_r_dev, (int)t_nr, stream);
  offloadEventRecord(event_compute_done[1], stream);
  if (lazy) {
    if (natural) {
      res_register_natural_lazy(rho_g_host, t_nr, rho_g_dev);
    } else {
      res_register_lazy(RES_RHO_G, rho_g_host, (size_t)t_ngpts, 2, rho_g_dev, samples);
    }
    res_register_lazy(RES_RHO_R, rho_r_host, t_nr, 1, rho_r_dev, samples);
    offloadStreamSynchronize(stream);
    res_stats.density++;
    res_stats.lazy_density++;
    res_stats.download_saved += sizeof(double) * (2 * (size_t)t_ngpts + t_nr);
    return 1;
  }
  // Sorted-G downloads overlap the inverse FFT. Natural-G downloads gather
  // through bounded staging on the compute stream, preserving host ordering.
  offloadStreamWaitEvent(stream_copy, event_compute_done[0]);
  if (natural) {
    res_natural_download(rho_g_dev, rho_g_host, t_map, t_nr);
  } else {
    offloadMemcpyAsyncDtoH(rho_g_host, rho_g_dev,
                           2 * sizeof(double) * (size_t)t_ngpts, stream_copy);
  }
  offloadStreamWaitEvent(stream_copy, event_compute_done[1]);
  offloadMemcpyAsyncDtoH(rho_r_host, rho_r_dev, sizeof(double) * t_nr,
                         stream_copy);
  offloadStreamSynchronize(stream_copy);
  offloadStreamSynchronize(stream);
  res_register(RES_RHO_G, rho_g_host, (size_t)t_ngpts, 2, 2, rho_g_dev);
  res[RES_RHO_G].natural_g = natural;
  res_register(RES_RHO_R, rho_r_host, t_nr, 1, 1, rho_r_dev);
  res_stats.density++;
  return 1;
}

int pw_gpu_res_density_c(const int nlevels, void *const *r_host, const int *npts,
                         const int *ngpts, const int *nmaps, const int *grid_id,
                         void *const *maps, void *const *gidx, const double *scale,
                         const int *t_npts, const int t_ngpts, const int t_nmaps,
                         const int t_grid_id, const int *t_map, double *rho_g_host,
                         double *rho_r_host, void *const *r_dev) {
  // The token is consumed by every attempt, including rejected ones.
  const bool lazy_allowed = res_density_token;
  res_density_token = false;
  const int done = res_density(lazy_allowed, nlevels, r_host, r_dev, npts, ngpts, nmaps,
                               grid_id, maps, gidx, scale, t_npts, t_ngpts,
                               t_nmaps, t_grid_id, t_map, rho_g_host, rho_r_host);
  protected_xc_buffer = false;
  return done;
}

/*******************************************************************************
 * \brief potential_pw2rs (pw_interp) on the device. v_host is the potential on
 *        the grid of level aux; r_host[aux] receives a copy of it (on the host,
 *        concurrently with the device work), every other level l
 *          r_l = rscale_l * Re FFT^-1[scatter(G_l)],  G_l(i) = G(gidx_l(i)),
 *          G = gather(fwd_scale FFT[v]).
 *        The resident V_H + dvol*Vxc (pw_gpu_res_vtot_commit_c) replaces the
 *        upload of v_host when it matches.
 ******************************************************************************/
static int res_potential(const double *v_host, const int nlevels,
                         void *const *r_host, const int *npts, const int *ngpts,
                         const int *nmaps, const int *grid_id, void *const *maps,
                         void *const *gidx, const double *rscale, const int aux,
                         const double fwd_scale, void **aux_dev) {
  if (!is_initialized || !opt_res_pot || !opt_map_cache ||
      omp_get_num_threads() != 1) {
    return 0;
  }
  if (v_host == NULL || nlevels < 1 || nlevels > PW_GPU_RES_MAX_LEVELS ||
      aux < 0 || aux >= nlevels) {
    return 0;
  }
  size_t nr[PW_GPU_RES_MAX_LEVELS], off[PW_GPU_RES_MAX_LEVELS];
  size_t nr_max = 0, need = 0;
  for (int l = 0; l < nlevels; l++) {
    if (!res_grid_points(npts + 3 * l, &nr[l]) || ngpts[l] <= 0 ||
        (size_t)ngpts[l] > nr[l] || (nmaps[l] != 1 && nmaps[l] != 2) ||
        grid_id[l] <= 0 || r_host[l] == NULL || maps[l] == NULL) {
      return 0;
    }
    if (l != aux && (gidx[l] == NULL || ngpts[l] > ngpts[aux])) {
      return 0;
    }
    nr_max = nr[l] > nr_max ? nr[l] : nr_max;
    if (l != aux) {
      need += 2 * (size_t)ngpts[l] > nr[l] ? 2 * (size_t)ngpts[l] : nr[l];
    }
  }
  offload_activate_chosen_device();
  if (!pw_gpu_ensure_buffer_c(2 * sizeof(double) * nr_max, 0, false) ||
      !pw_gpu_ensure_gbuf_c(2 * sizeof(double) * (size_t)ngpts[aux])) {
    res_stats.potential_fail++;
    return 0;
  }
  const double *src = res_lookup(RES_VXC, v_host, nr[aux], true);
  // A borrowed output must live in the resident XC allocation, never scratch
  // that the interpolation itself or a returned PW pool could overwrite.
  if (aux_dev != NULL && src == NULL) return 0;
  protected_xc_buffer = src != NULL; // grid cache uploads must not release it
  size_t start = 0;
  if (src == NULL) {
    start = nr[aux];
  }
  if (start + need > 2 * nr_max) {
    return 0; // the coarse levels must fit next to the uploaded potential
  }
  size_t o = start;
  for (int l = 0; l < nlevels; l++) {
    off[l] = o;
    if (l != aux) {
      o += 2 * (size_t)ngpts[l] > nr[l] ? 2 * (size_t)ngpts[l] : nr[l];
    }
  }
  if (src == NULL) {
    offloadMemcpyAsyncHtoD(buffer_dev_1, v_host, sizeof(double) * nr[aux], stream);
    src = buffer_dev_1;
  }
  grid_cache_entry *a =
      grid_cache_lookup_or_upload(grid_id[aux], maps[aux], ngpts[aux], nmaps[aux]);
  if (a == NULL) {
    offloadStreamSynchronize(stream);
    res_stats.potential_fail++;
    return 0;
  }
  pw_gpu_launch_real_to_complex(src, buffer_dev_2, (int)nr[aux], stream);
  fft_3d(OFFLOAD_FFT_FORWARD, npts[3 * aux + 2], npts[3 * aux + 1], npts[3 * aux],
         buffer_dev_2);
  pw_gpu_launch_gather(gbuf_dev, buffer_dev_2, fwd_scale, ngpts[aux], a->map_dev,
                       stream);
  for (int l = 0; l < nlevels; l++) {
    if (l == aux) continue;
    grid_cache_entry *e =
        grid_cache_lookup_or_upload(grid_id[l], maps[l], ngpts[l], nmaps[l]);
    const int *gidx_dev = grid_cache_get_gidx(e, gidx[l], ngpts[aux], NULL);
    if (e == NULL || gidx_dev == NULL) {
      offloadStreamSynchronize(stream);
      res_stats.potential_fail++;
      return 0;
    }
    double *buf = buffer_dev_1 + off[l];
    pw_gpu_launch_remap_copy(gbuf_dev, buf, gidx_dev, (size_t)ngpts[l], 2, 2, 0,
                             1.0, stream);
    offloadMemsetAsync(buffer_dev_2, 0, 2 * sizeof(double) * nr[l], stream);
    pw_gpu_launch_scatter(buffer_dev_2, buf, 1.0, ngpts[l], nmaps[l], e->map_dev,
                          stream);
    fft_3d(OFFLOAD_FFT_INVERSE, npts[3 * l + 2], npts[3 * l + 1], npts[3 * l],
           buffer_dev_2);
    pw_gpu_launch_complex_to_real(buffer_dev_2, buf, (int)nr[l], stream);
    pw_gpu_launch_copy(buf, buf, nr[l], nr[l], 1, 1, rscale[l], stream);
  }
  // The auxiliary level also comes from the device when the host AXPY was
  // skipped. Do not copy the stale host Vxc into the real-space grid.
  if (aux_dev != NULL) {
    const grid_buffer_descriptor source = {(void *)src, sizeof(double) * nr[aux]};
    grid_buffer_descriptor view;
    if (!grid_pw_bridge(&source, source.nbytes, &view)) return 0;
    *aux_dev = view.dev_ptr;
    res_stats.download_saved += source.nbytes;
  } else if (res[RES_VXC].dev_authoritative && res[RES_VXC].host == v_host) {
    offloadMemcpyAsyncDtoH(r_host[aux], src, sizeof(double) * nr[aux], stream);
    if (r_host[aux] == v_host) res[RES_VXC].dev_authoritative = false;
  } else {
    res_host_copy(r_host[aux], v_host, nr[aux]);
  }
  for (int l = 0; l < nlevels; l++) {
    if (l != aux) {
      offloadMemcpyAsyncDtoH(r_host[l], buffer_dev_1 + off[l],
                             sizeof(double) * nr[l], stream);
    }
  }
  offloadStreamSynchronize(stream);
  res_stats.potential++;
  return 1;
}

int pw_gpu_res_potential_c(const double *v_host, const int nlevels,
                           void *const *r_host, const int *npts, const int *ngpts,
                           const int *nmaps, const int *grid_id, void *const *maps,
                           void *const *gidx, const double *rscale, const int aux,
                           const double fwd_scale, void **aux_dev) {
  if (aux_dev != NULL) *aux_dev = NULL;
  const int done = res_potential(v_host, nlevels, r_host, npts, ngpts, nmaps,
                                 grid_id, maps, gidx, rscale, aux, fwd_scale, aux_dev);
  protected_xc_buffer = false;
  if (!done) pw_gpu_res_ensure_host_c(RES_VXC, (void *)v_host);
  return done;
}

/*******************************************************************************
 * \brief Before the host executes pw_axpy(v_h, vxc, 1, dvol): forms
 *          vxc_dev = dvol*vxc_dev + v_h_dev
 *        with the same kernel and rounding in the resident Vxc grid and keeps
 *        its fingerprint, which pw_gpu_res_vtot_commit_c compares with the
 *        host result. Returns 0 (nothing pending) unless Vxc is resident.
 ******************************************************************************/
int pw_gpu_res_vtot_prepare_c(const double *vh_host, const double *vxc_host,
                              const size_t n, const double dvol) {
  if (!is_initialized || !opt_res_pot || omp_get_num_threads() != 1 || n == 0 ||
      vh_host == NULL || n > (size_t)INT_MAX) {
    return 0;
  }
  offload_activate_chosen_device();
  res_slot *s = &res[RES_VXC];
  protected_xc_buffer = true;
  double *samples = res_small(pw_res_sample_count(n));
  protected_xc_buffer = false;
  double *vxc = samples == NULL ? NULL : res_lookup(RES_VXC, vxc_host, n, false);
  if (vxc == NULL || !res_sample_buffer(s, pw_res_sample_count(n))) {
    res_stats.vtot_fail++;
    return 0;
  }
  res_pinned |= 1u << RES_V_H_R;
  const double *vh = res_lookup(RES_V_H_R, vh_host, n, false);
  if (vh == NULL) {
    protected_xc_buffer = true;
    const int ok = pw_gpu_ensure_buffer_c(sizeof(double) * n, 0, false);
    protected_xc_buffer = false;
    if (!ok || s->epoch != xc_epoch) {
      res_pinned = 0;
      if (s->epoch == xc_epoch) res_sync(RES_VXC); // a lazy Vxc is still intact
      res_invalidate(RES_VXC);
      res_stats.vtot_fail++;
      return 0;
    }
    offloadMemcpyAsyncHtoD(buffer_dev_1, vh_host, sizeof(double) * n, stream);
    vh = buffer_dev_1;
  }
  pw_gpu_launch_axpy(vh, vxc, n, n, 1, 1, 1.0, dvol, stream);
  pw_gpu_launch_res_sample(vxc, samples, n, 1, stream);
  offloadMemcpyAsyncDtoH(s->samples, samples,
                         sizeof(double) * pw_res_sample_count(n), stream);
  offloadStreamSynchronize(stream);
  res_pinned = 0;
  s->state = RES_PENDING;
  s->vtot = true;
  return 1;
}

// Opt into the device result instead of the legacy host AXPY/commit pair.
int pw_gpu_res_vtot_skip_host_c(void) {
  if (!is_initialized || !opt_res_pot || omp_get_num_threads() != 1) return 0;
  res_slot *s = &res[RES_VXC];
  if (s->state == RES_VALID && s->vtot && s->dev_authoritative && s->epoch == xc_epoch) return 1;
  if (s->state != RES_PENDING || !s->vtot || s->epoch != xc_epoch || s->dev == NULL) return 0;
  s->state = RES_VALID;
  s->dev_authoritative = true;
  res_stats.vtot++;
  return 1;
}

/*******************************************************************************
 * \brief After the host pw_axpy: the device sum becomes resident if the host
 *        result agrees at every fingerprint position.
 ******************************************************************************/
int pw_gpu_res_vtot_commit_c(const double *vtot_host, const size_t n) {
  if (!is_initialized) {
    return 0;
  }
  res_slot *s = &res[RES_VXC];
  bool ok = s->state == RES_PENDING && s->vtot && s->host == vtot_host &&
            s->n == n && s->epoch == xc_epoch && s->dev != NULL;
  ok = ok && res_samples_match(s, vtot_host);
  ok = ok && (!opt_res_verify || res_verify_full(s, vtot_host));
  if (!ok) {
    res_invalidate(RES_VXC);
    res_stats.vtot_fail++;
    return 0;
  }
  s->state = RES_VALID;
  res_stats.vtot++;
  return 1;
}

/*******************************************************************************
 * \brief Hartree/DDAPC chain, driven from cp_ddapc.F in four steps:
 *          begin : rho_tot = rho_core + rho_g        (buffer_dev_2)
 *                  prefix of rho_tot -> host         (DDAPC charge fit)
 *                  v_H = rho_tot * influence         (gbuf)
 *          modify: G=0 term and DDAPC correction of v_H on the device
 *          energy: Re sum conj(rho_tot) v_H          (deterministic sum)
 *          finish: v_H_r = dvol * Re FFT^-1[scatter(v_H)] -> host, resident
 *        Natural full boxes omit the final scatter; only the DDAPC prefix
 *        uses indirect accesses, and core/influence conversions are cached.
 *        The steps share the scratch buffers. Any other PW GPU call in between
 *        changes scratch_epoch, after which the remaining steps return 0 and
 *        the caller recomputes everything on the original path.
 ******************************************************************************/
static int res_hartree_fail(void) {
  if (is_initialized) {
    offloadStreamSynchronize(stream);
  }
  res_hartree.pending = false;
  res_pinned = 0;
  res_stats.hartree_fail++;
  return 0;
}

static bool res_hartree_ready(void) {
  return is_initialized && res_hartree.pending &&
         res_hartree.epoch == scratch_epoch && omp_get_num_threads() == 1;
}

int pw_gpu_res_hartree_begin_c(const double *core_host, const double *rho_g_host,
                               const double *infl_host, double *rho_tot_host,
                               const size_t ngpts, const size_t nprefix,
                               const int *npts) {
  res_hartree.pending = false;
  if (!is_initialized || !opt_res_hartree || !opt_map_cache ||
      omp_get_num_threads() != 1) {
    return 0;
  }
  size_t nr;
  if (core_host == NULL || rho_g_host == NULL || infl_host == NULL ||
      rho_tot_host == NULL || npts == NULL || !res_grid_points(npts, &nr) ||
      ngpts == 0 || ngpts > nr || nprefix == 0 || nprefix > ngpts) {
    return 0;
  }
  offload_activate_chosen_device();
  const size_t zbytes = 2 * sizeof(double) * ngpts;
  const bool natural = opt_res_natural_g && ngpts == nr &&
      res[RES_RHO_G].state == RES_VALID && res[RES_RHO_G].natural_g &&
      res[RES_RHO_G].host == rho_g_host && res[RES_RHO_G].n == ngpts &&
      res_natural.n == nr && !memcmp(res_natural.npts, npts, sizeof(res_natural.npts));
  protected_xc_buffer = natural;
  const bool fits = pw_gpu_ensure_buffer_c(2 * sizeof(double) * nr, 0, false) &&
                    pw_gpu_ensure_gbuf_c(zbytes);
  protected_xc_buffer = false;
  if (!fits) {
    res_stats.hartree_fail++;
    return 0;
  }
  double *rho_tot = buffer_dev_2, *v_h = gbuf_dev;

  // rho_tot = pw_copy(rho_core); pw_axpy(rho_g, rho_tot)
  double *core = res_lookup_g(RES_RHO_CORE, core_host, ngpts, natural);
  if (core == NULL && res_ensure_owned(RES_RHO_CORE, zbytes)) {
    core = res[RES_RHO_CORE].dev;
    if (natural) {
      if (!res_natural_upload(core, core_host, 2)) return res_hartree_fail();
    } else {
      offloadMemcpyAsyncHtoD(core, core_host, zbytes, stream);
    }
    res_register(RES_RHO_CORE, core_host, ngpts, 2, 2, core);
    res[RES_RHO_CORE].natural_g = natural;
  }
  res_pinned |= 1u << RES_RHO_CORE;
  if (core != NULL) {
    pw_gpu_launch_copy(core, rho_tot, ngpts, ngpts, 2, 2, 1.0, stream);
  } else if (natural) {
    if (!res_natural_upload(rho_tot, core_host, 2)) return res_hartree_fail();
  } else {
    offloadMemcpyAsyncHtoD(rho_tot, core_host, zbytes, stream);
  }
  const double *rg = res_lookup_g(RES_RHO_G, rho_g_host, ngpts, natural);
  if (rg == NULL) {
    if (natural) {
      if (!res_natural_upload(v_h, rho_g_host, 2)) return res_hartree_fail();
    } else {
      offloadMemcpyAsyncHtoD(v_h, rho_g_host, zbytes, stream);
    }
    rg = v_h;
  }
  pw_gpu_launch_axpy(rg, rho_tot, ngpts, ngpts, 2, 2, 1.0, 1.0, stream);

  // v_H = rho_tot * influence. A real influence function is kept compactly.
  int width = 1;
  double *infl = res_lookup_g(RES_INFL, infl_host, ngpts, natural);
  if (natural) {
    if (infl != NULL) {
      width = res[RES_INFL].width;
    } else {
      width = res_imag_zero(infl_host, ngpts) ? 1 : 2;
      const bool resident = res_ensure_owned(RES_INFL, sizeof(double) * ngpts * (size_t)width);
      infl = resident ? res[RES_INFL].dev : buffer_dev_1;
      if (!res_natural_upload(infl, infl_host, width)) return res_hartree_fail();
      if (resident) {
        res_register(RES_INFL, infl_host, ngpts, width, 2, infl);
        res[RES_INFL].natural_g = true;
      }
    }
  } else if (infl == NULL) {
    offloadMemcpyAsyncHtoD(buffer_dev_1, infl_host, zbytes, stream);
    bool compact = res_ensure_owned(RES_INFL, sizeof(double) * ngpts);
    if (compact && !res_imag_zero(infl_host, ngpts)) {
      res_free_owned(RES_INFL);
      compact = false;
    }
    if (compact) {
      infl = res[RES_INFL].dev;
      pw_gpu_launch_copy(buffer_dev_1, infl, ngpts, ngpts, 2, 1, 1.0, stream);
      res_register(RES_INFL, infl_host, ngpts, 1, 2, infl);
    } else {
      infl = buffer_dev_1;
      width = 2;
    }
  }
  if (natural) {
    pw_gpu_launch_res_natural_cmul(v_h, rho_tot, infl, width, nr, stream);
    res_natural_download(rho_tot, rho_tot_host, res_natural.map, nprefix);
  } else {
    pw_gpu_launch_res_cmul(v_h, rho_tot, infl, width, ngpts, stream);
    offloadMemcpyAsyncDtoH(rho_tot_host, rho_tot, 2 * sizeof(double) * nprefix, stream);
  }
  offloadStreamSynchronize(stream);
  res_pinned = 0;
  res_hartree.pending = true;
  res_hartree.epoch = scratch_epoch;
  res_hartree.ngpts = ngpts;
  res_hartree.natural_g = natural;
  memcpy(res_hartree.npts, npts, sizeof(res_hartree.npts));
  return 1;
}

int pw_gpu_res_hartree_modify_c(const double g0_delta, const int have_g0,
                                const double *g, const double *gsq,
                                const double *r, const double *radii,
                                const double *cv, const size_t first,
                                const size_t n, const size_t natoms,
                                const size_t nradii, const double gcut2,
                                const double vol) {
  if (!res_hartree_ready()) {
    return res_hartree_fail();
  }
  // The sorted-G loop of modify_hartree_pot exits at the first |G|^2 > GCUT^2.
  size_t active = 0;
  while (active < n && !(gsq[active] > gcut2)) {
    active++;
  }
  if (first > res_hartree.ngpts || active > res_hartree.ngpts - first ||
      (nradii != 0 && natoms > SIZE_MAX / nradii / 8)) {
    return res_hartree_fail();
  }
  const size_t counts[6] = {2, 3 * active, active, 3 * natoms, nradii,
                            natoms * nradii};
  size_t total = 0;
  for (int i = 0; i < 6; i++) {
    total += counts[i];
  }
  const size_t params = total;
  if (res_hartree.natural_g) {
    total += 2 * active + (active * sizeof(int) + sizeof(double) - 1) / sizeof(double);
  }
  double *small = res_small(total);
  if (small == NULL) {
    return res_hartree_fail();
  }
  static double g0_term[2];
  g0_term[0] = g0_delta;
  g0_term[1] = 0.0; // complex + real in Fortran adds (delta, 0)
  const double *host[6] = {g0_term, g, gsq, r, radii, cv};
  double *dev[6];
  size_t offset = 0;
  for (int i = 0; i < 6; i++) {
    dev[i] = small + offset;
    offset += counts[i];
    if (counts[i] != 0 && (i == 0 || i > 2 || active != 0)) {
      offloadMemcpyAsyncHtoD(dev[i], host[i], sizeof(double) * counts[i], stream);
    }
  }
  if (have_g0) {
    const size_t g0 = res_hartree.natural_g ? (size_t)res_natural.map[0] : 0;
    pw_gpu_launch_axpy(dev[0], gbuf_dev + 2 * g0, 1, 1, 2, 2, 1.0, 1.0, stream);
  }
  if (active != 0) {
    double *v = res_hartree.natural_g ? small + params : gbuf_dev + 2 * first;
    int *coords = res_hartree.natural_g ? (int *)(small + params + 2 * active) : NULL;
    if (res_hartree.natural_g) {
      offloadMemcpyAsyncHtoD(coords, res_natural.map + first, active * sizeof(int), stream);
      pw_gpu_launch_res_natural_extract_prefix(gbuf_dev, v, coords, active, stream);
    }
    pw_gpu_launch_modify_hartree(v, dev[1], dev[2], dev[3],
                                 dev[4], dev[5], active, natoms, nradii, gcut2,
                                 vol, stream);
    if (res_hartree.natural_g)
      pw_gpu_launch_scatter(gbuf_dev, v, 1.0, (int)active, 1, coords, stream);
  }
  // The host arguments may be released by the caller after the return.
  offloadStreamSynchronize(stream);
  return 1;
}

int pw_gpu_res_hartree_energy_c(double *dot) {
  if (!res_hartree_ready() || dot == NULL) {
    return res_hartree_fail();
  }
  const size_t n = res_hartree.ngpts;
  double *terms = buffer_dev_1, *partial = buffer_dev_1 + n;
  if (res_hartree.natural_g) {
    pw_gpu_launch_res_natural_cdot_terms(buffer_dev_2, gbuf_dev, terms, n, stream);
  } else {
    pw_gpu_launch_res_cdot_terms(buffer_dev_2, gbuf_dev, terms, n, stream);
  }
  size_t remaining = n;
  double *sum_in = terms, *sum_out = partial;
  do {
    pw_gpu_launch_xc_reduce(sum_in, sum_out, remaining, stream);
    remaining = pw_gpu_xc_reduce_size(remaining);
    double *swap = sum_in;
    sum_in = sum_out;
    sum_out = swap;
  } while (remaining > 1);
  offloadMemcpyAsyncDtoH(dot, sum_in, sizeof(double), stream);
  offloadStreamSynchronize(stream);
  return 1;
}

int pw_gpu_res_hartree_finish_c(double *v_h_r_host, const int *npts,
                                const int ngpts, const int nmaps,
                                const int grid_id, const int *ghatmap,
                                const double dvol) {
  size_t nr;
  if (!res_hartree_ready() || v_h_r_host == NULL || ghatmap == NULL || npts == NULL ||
      !res_grid_points(npts, &nr) || ngpts <= 0 ||
      (size_t)ngpts != res_hartree.ngpts || (nmaps != 1 && nmaps != 2) ||
      grid_id <= 0 || memcmp(npts, res_hartree.npts, sizeof(res_hartree.npts))) {
    return res_hartree_fail();
  }
  const bool natural = res_hartree.natural_g;
  if (natural && (nmaps != 1 || grid_id != res_natural.grid_id || nr != res_natural.n)) {
    return res_hartree_fail();
  }
  grid_cache_entry *e = natural ? NULL :
      grid_cache_lookup_or_upload(grid_id, ghatmap, ngpts, nmaps);
  if ((!natural && e == NULL) || !res_hartree_ready()) return res_hartree_fail();
  double *dst = buffer_dev_1;
  protected_grid_entry = e;
  const bool resident = opt_res_pot && res_ensure_owned(RES_V_H_R, sizeof(double) * nr);
  protected_grid_entry = NULL;
  if (resident) {
    dst = res[RES_V_H_R].dev;
    // A live V_H of another host array is written back before it is replaced.
    if (res[RES_V_H_R].host != v_h_r_host) res_sync(RES_V_H_R);
  }
  // Inside the lazy window V_H stays on the device for V_TOT; the host copy is
  // written when a host consumer asks for it or at the end of the SCF.
  bool lazy = resident && res_lazy_grant && opt_res_lazy_pot;
  double *samples = NULL;
  if (lazy) {
    // Reclaiming memory for the fingerprint scratch must keep the map and V_H.
    res_pinned |= 1u << RES_V_H_R;
    protected_grid_entry = e;
    samples = res_small(pw_res_sample_count(nr));
    protected_grid_entry = NULL;
    res_pinned &= ~(1u << RES_V_H_R);
    lazy = samples != NULL && res_sample_buffer(&res[RES_V_H_R], pw_res_sample_count(nr));
  }
  double *fft_input = natural ? gbuf_dev : buffer_dev_2;
  if (!natural) {
    offloadMemsetAsync(buffer_dev_2, 0, 2 * sizeof(double) * nr, stream);
    pw_gpu_launch_scatter(buffer_dev_2, gbuf_dev, 1.0, ngpts, nmaps, e->map_dev, stream);
  }
  fft_3d(OFFLOAD_FFT_INVERSE, npts[2], npts[1], npts[0], fft_input);
  pw_gpu_launch_complex_to_real(fft_input, dst, (int)nr, stream);
  pw_gpu_launch_copy(dst, dst, nr, nr, 1, 1, dvol, stream); // pw_scale
  if (lazy) {
    res_register_lazy(RES_V_H_R, v_h_r_host, nr, 1, dst, samples);
    offloadStreamSynchronize(stream);
    res_stats.lazy_vhr++;
    res_stats.download_saved += sizeof(double) * nr;
  } else {
    offloadMemcpyAsyncDtoH(v_h_r_host, dst, sizeof(double) * nr, stream);
    offloadStreamSynchronize(stream);
    if (resident) {
      res_register(RES_V_H_R, v_h_r_host, nr, 1, 1, dst);
    }
  }
  res_hartree.pending = false;
  res_stats.hartree++;
  return 1;
}

void pw_gpu_res_hartree_cancel_c(void) {
  if (is_initialized && res_hartree.pending) {
    offloadStreamSynchronize(stream);
  }
  res_hartree.pending = false;
}

/*******************************************************************************
 * \brief   Performs a (double precision complex) FFT, followed by a (double
 *          precision complex) gather, on the GPU.
 *          grid_id > 0 enables the device-side g_hatmap cache.
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
void pw_gpu_cfffg_cached(const double *din, double *zout, const int *ghatmap,
                         const int *npts, const int ngpts, const int nmaps,
                         const double scale, const int grid_id) {
  // Check inputs.
  assert(omp_get_num_threads() == 1);
  const int nrpts = npts[0] * npts[1] * npts[2];
  assert(ngpts <= nrpts);
  if (nrpts == 0 || ngpts == 0) {
    return; // Nothing to do.
  }

  // Allocate device memory.
  offload_activate_chosen_device();
  const size_t buffer_size = 2 * sizeof(double) * nrpts;
  const size_t map_size = sizeof(int) * ngpts;
  const int *map_dev = NULL;
  if (!pw_gpu_ensure_buffer_c(buffer_size, 0, false) ||
      (map_dev = get_map_dev(grid_id, ghatmap, ngpts, nmaps, map_size)) == NULL) {
    host_cfffg(din, zout, ghatmap, npts, ngpts, scale);
    return;
  }

  // Upload REAL input and convert to COMPLEX on device.
  offloadMemcpyAsyncHtoD(buffer_dev_1, din, buffer_size / 2, stream);
  pw_gpu_launch_real_to_complex(buffer_dev_1, buffer_dev_2, nrpts, stream);

  // Run FFT on the device.
  fft_3d(OFFLOAD_FFT_FORWARD, npts[2], npts[1], npts[0], buffer_dev_2);

  // Get map (cached or uploaded) and run gather on the device.
  pw_gpu_launch_gather(buffer_dev_1, buffer_dev_2, scale, ngpts, map_dev,
                       stream);

  // Download COMPLEX results to host.
  offloadMemcpyAsyncDtoH(zout, buffer_dev_1, 2 * sizeof(double) * ngpts,
                         stream);
  offloadStreamSynchronize(stream);
}

/*******************************************************************************
 * \brief   Legacy entry point of pw_gpu_cfffg_cached without map cache.
 ******************************************************************************/
void pw_gpu_cfffg(const double *din, double *zout, const int *ghatmap,
                  const int *npts, const int ngpts, const double scale) {
  pw_gpu_cfffg_cached(din, zout, ghatmap, npts, ngpts, 1, scale, 0);
}

/*******************************************************************************
 * \brief   Performs a (double precision complex) scatter, followed by a
 *          (double precision complex) FFT, on the GPU.
 *          grid_id > 0 enables the device-side g_hatmap cache.
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
void pw_gpu_sfffc_cached(const double *zin, double *dout, const int *ghatmap,
                         const int *npts, const int ngpts, const int nmaps,
                         const double scale, const int grid_id) {
  // Check inputs.
  assert(omp_get_num_threads() == 1);
  const int nrpts = npts[0] * npts[1] * npts[2];
  assert(ngpts <= nrpts);
  if (nrpts == 0 || ngpts == 0) {
    return; // Nothing to do.
  }

  // Allocate device memory.
  offload_activate_chosen_device();
  const size_t buffer_size = 2 * sizeof(double) * nrpts;
  const size_t map_size = sizeof(int) * nmaps * ngpts;
  const int *map_dev = NULL;
  if (!pw_gpu_ensure_buffer_c(buffer_size, 0, false) ||
      (map_dev = get_map_dev(grid_id, ghatmap, ngpts, nmaps, map_size)) == NULL) {
    host_sfffc(zin, dout, ghatmap, npts, ngpts, nmaps, scale);
    return;
  }

  // Upload COMPLEX inputs to device.
  offloadMemcpyAsyncHtoD(buffer_dev_1, zin, 2 * sizeof(double) * ngpts, stream);

  // Get map (cached or uploaded) and run scatter on the device.
  offloadMemsetAsync(buffer_dev_2, 0, buffer_size, stream);
  pw_gpu_launch_scatter(buffer_dev_2, buffer_dev_1, scale, ngpts, nmaps,
                        map_dev, stream);

  // Run FFT on the device.
  fft_3d(OFFLOAD_FFT_INVERSE, npts[2], npts[1], npts[0], buffer_dev_2);

  // Convert COMPLEX results to REAL and download to host.
  pw_gpu_launch_complex_to_real(buffer_dev_2, buffer_dev_1, nrpts, stream);
  offloadMemcpyAsyncDtoH(dout, buffer_dev_1, buffer_size / 2, stream);
  offloadStreamSynchronize(stream);
}

/*******************************************************************************
 * \brief   Legacy entry point of pw_gpu_sfffc_cached without map cache.
 ******************************************************************************/
void pw_gpu_sfffc(const double *zin, double *dout, const int *ghatmap,
                  const int *npts, const int ngpts, const int nmaps,
                  const double scale) {
  pw_gpu_sfffc_cached(zin, dout, ghatmap, npts, ngpts, nmaps, scale, 0);
}

/*******************************************************************************
 * \brief   Fused XC gradient: dout_d = FFT^-1[ scatter( (0,1) g_d zin ) ] for
 *          d = 1,2,3. zin is uploaded only once; the derivative is applied
 *          on the device. Bitwise identical to
 *            3 x { pw_copy, pw_derive (CPU), pw_gpu_sfffc(scale=1) }.
 *          dout_1..3 must each hold npts[0]*npts[1]*npts[2] doubles.
 ******************************************************************************/
void pw_gpu_sfffc_grad3(const double *zin, double *dout_1, double *dout_2,
                        double *dout_3, const int *ghatmap, const double *g,
                        const int *npts, const int ngpts, const int nmaps,
                        const int grid_id) {
  assert(omp_get_num_threads() == 1);
  assert(grid_id > 0);
  const int nrpts = npts[0] * npts[1] * npts[2];
  assert(ngpts <= nrpts);
  double *dout[3] = {dout_1, dout_2, dout_3};
  if (nrpts == 0) {
    return; // Nothing to do.
  }
  if (ngpts == 0) {
    // Unfused path: pw_zero(gradient) followed by a no-op transfer.
    for (int d = 0; d < 3; d++) {
      memset(dout[d], 0, sizeof(double) * nrpts);
    }
    return;
  }

  // CPU XC construction can reach this host-input path with a lazy density.
  pw_gpu_res_ensure_host_c(RES_RHO_G, (void *)zin);
  offload_activate_chosen_device();
  const size_t buffer_size = 2 * sizeof(double) * nrpts;
  const size_t gbuf_size = 2 * sizeof(double) * ngpts;
  grid_cache_entry *e = NULL;
  const double *g_dev = NULL;
  if (!pw_gpu_ensure_buffer_c(buffer_size, 0, false) ||
      !pw_gpu_ensure_gbuf_c(gbuf_size) ||
      (e = grid_cache_lookup_or_upload(grid_id, ghatmap, ngpts, nmaps)) == NULL ||
      (g_dev = grid_cache_get_g(e, g)) == NULL) {
    host_grad3(zin, dout, ghatmap, g, npts, ngpts, nmaps);
    return;
  }

  protected_grid_entry = e;
  pw_gpu_ensure_buffer_c(buffer_size, 0, true);
  protected_grid_entry = NULL;
  const bool overlap = (allocated_buffer_size_b >= buffer_size);

  // Upload COMPLEX input once.
  offloadMemcpyAsyncHtoD(gbuf_dev, zin, gbuf_size, stream);

  for (int d = 0; d < 3; d++) {
    const int slot = d % 2;
    double *real = overlap && slot ? buffer_dev_1_b : buffer_dev_1;
    double *complex = overlap && slot ? buffer_dev_2_b : buffer_dev_2;
    if (overlap && d >= 2) {
      offloadStreamWaitEvent(stream, event_copy_done[slot]);
    }
    offloadMemsetAsync(complex, 0, buffer_size, stream);
    pw_gpu_launch_scatter_deriv(complex, gbuf_dev, g_dev, d, 1.0, ngpts,
                                nmaps, e->map_dev, stream);
    fft_3d(OFFLOAD_FFT_INVERSE, npts[2], npts[1], npts[0], complex);
    pw_gpu_launch_complex_to_real(complex, real, nrpts, stream);
    if (overlap) {
      offloadEventRecord(event_compute_done[slot], stream);
      offloadStreamWaitEvent(stream_copy, event_compute_done[slot]);
      offloadMemcpyAsyncDtoH(dout[d], real, buffer_size / 2, stream_copy);
      offloadEventRecord(event_copy_done[slot], stream_copy);
    } else {
      offloadMemcpyAsyncDtoH(dout[d], real, buffer_size / 2, stream);
    }
  }
  offloadStreamSynchronize(stream);
  offloadStreamSynchronize(stream_copy);
}

/*******************************************************************************
 * \brief   Fused XC divergence:
 *            dout = FFT^-1[ scatter( sum_d (0,1) g_d gather(scale FFT[din_d]) ) ]
 *          The G-space accumulator stays on the device; only the final real
 *          space result is downloaded. Bitwise identical to
 *            pw_zero(vxc_g); 3 x { pw_gpu_cfffg, pw_derive, pw_axpy (CPU) };
 *            pw_gpu_sfffc(vxc_g, scale=1).
 *          dout may alias din_1 (all inputs are consumed before the download).
 ******************************************************************************/
void pw_gpu_cfffg_div3(const double *din_1, const double *din_2,
                       const double *din_3, double *dout, const int *ghatmap,
                       const double *g, const int *npts, const int ngpts,
                       const int nmaps, const double scale,
                       const int grid_id) {
  assert(omp_get_num_threads() == 1);
  assert(grid_id > 0);
  const int nrpts = npts[0] * npts[1] * npts[2];
  assert(ngpts <= nrpts);
  const double *din[3] = {din_1, din_2, din_3};
  if (nrpts == 0) {
    return; // Nothing to do.
  }
  if (ngpts == 0) {
    // Unfused path: transfers are no-ops, dout keeps its content.
    return;
  }

  offload_activate_chosen_device();
  const size_t buffer_size = 2 * sizeof(double) * nrpts;
  const size_t gbuf_size = 2 * sizeof(double) * ngpts;
  grid_cache_entry *e = NULL;
  const double *g_dev = NULL;
  if (!pw_gpu_ensure_buffer_c(buffer_size, 0, false) ||
      !pw_gpu_ensure_gbuf_c(gbuf_size) ||
      (e = grid_cache_lookup_or_upload(grid_id, ghatmap, ngpts, nmaps)) == NULL ||
      (g_dev = grid_cache_get_g(e, g)) == NULL) {
    host_div3(din, dout, ghatmap, g, npts, ngpts, nmaps, scale);
    return;
  }

  protected_grid_entry = e;
  pw_gpu_ensure_buffer_c(buffer_size, 0, true);
  protected_grid_entry = NULL;
  const bool overlap = (allocated_buffer_size_b >= buffer_size);
  const offloadStream_t upload_stream = overlap ? stream_copy : stream;

  // pw_zero(vxc_g)
  offloadMemsetAsync(gbuf_dev, 0, gbuf_size, stream);

  for (int d = 0; d < 3; d++) {
    const int slot = d % 2;
    double *real = overlap && slot ? buffer_dev_1_b : buffer_dev_1;
    double *complex = overlap && slot ? buffer_dev_2_b : buffer_dev_2;
    if (overlap && d >= 2) {
      offloadStreamWaitEvent(stream_copy, event_compute_done[slot]);
    }
    offloadMemcpyAsyncHtoD(real, din[d], buffer_size / 2, upload_stream);
    if (overlap) {
      offloadEventRecord(event_copy_done[slot], stream_copy);
      offloadStreamWaitEvent(stream, event_copy_done[slot]);
    }
    pw_gpu_launch_real_to_complex(real, complex, nrpts, stream);
    fft_3d(OFFLOAD_FFT_FORWARD, npts[2], npts[1], npts[0], complex);
    // Keep all three additions on the compute stream in the original order.
    pw_gpu_launch_gather_deriv_acc(gbuf_dev, complex, g_dev, d, scale,
                                   ngpts, e->map_dev, stream);
    if (overlap) {
      offloadEventRecord(event_compute_done[slot], stream);
    }
  }

  // pw_transfer(vxc_g, out): scatter with scale 1, inverse FFT, real part.
  offloadMemsetAsync(buffer_dev_2, 0, buffer_size, stream);
  pw_gpu_launch_scatter(buffer_dev_2, gbuf_dev, 1.0, ngpts, nmaps, e->map_dev,
                        stream);
  fft_3d(OFFLOAD_FFT_INVERSE, npts[2], npts[1], npts[0], buffer_dev_2);
  pw_gpu_launch_complex_to_real(buffer_dev_2, buffer_dev_1, nrpts, stream);
  offloadMemcpyAsyncDtoH(dout, buffer_dev_1, buffer_size / 2, stream);
  offloadStreamSynchronize(stream);
  offloadStreamSynchronize(stream_copy);
}

/*******************************************************************************
 * \brief   Fused spherical G-space filter of a real space grid (in place):
 *            data = FFT^-1[ scatter( gather(scale FFT[data]) ) ]
 *          Bitwise identical to pw_gpu_cfffg(scale) + pw_gpu_sfffc(scale=1),
 *          without the round trip of the G-space array through the host.
 ******************************************************************************/
void pw_gpu_filter_r3d(double *data, const int *ghatmap, const int *npts,
                       const int ngpts, const int nmaps, const double scale,
                       const int grid_id) {
  assert(omp_get_num_threads() == 1);
  assert(grid_id > 0);
  const int nrpts = npts[0] * npts[1] * npts[2];
  assert(ngpts <= nrpts);
  if (nrpts == 0 || ngpts == 0) {
    return; // Nothing to do (both unfused transfers would be no-ops).
  }

  offload_activate_chosen_device();
  const size_t buffer_size = 2 * sizeof(double) * nrpts;
  grid_cache_entry *e = NULL;
  if (!pw_gpu_ensure_buffer_c(buffer_size, 0, false) ||
      (e = grid_cache_lookup_or_upload(grid_id, ghatmap, ngpts, nmaps)) == NULL) {
    host_filter(data, ghatmap, npts, ngpts, nmaps, scale);
    return;
  }

  offloadMemcpyAsyncHtoD(buffer_dev_1, data, buffer_size / 2, stream);
  pw_gpu_launch_real_to_complex(buffer_dev_1, buffer_dev_2, nrpts, stream);
  fft_3d(OFFLOAD_FFT_FORWARD, npts[2], npts[1], npts[0], buffer_dev_2);
  pw_gpu_launch_gather(buffer_dev_1, buffer_dev_2, scale, ngpts, e->map_dev,
                       stream);
  offloadMemsetAsync(buffer_dev_2, 0, buffer_size, stream);
  pw_gpu_launch_scatter(buffer_dev_2, buffer_dev_1, 1.0, ngpts, nmaps,
                        e->map_dev, stream);
  fft_3d(OFFLOAD_FFT_INVERSE, npts[2], npts[1], npts[0], buffer_dev_2);
  pw_gpu_launch_complex_to_real(buffer_dev_2, buffer_dev_1, nrpts, stream);
  offloadMemcpyAsyncDtoH(data, buffer_dev_1, buffer_size / 2, stream);
  offloadStreamSynchronize(stream);
}

/*******************************************************************************
 * \brief   Performs a (double to complex double) blow-up and a (double
 *          precision complex) 2D-FFT on the GPU.
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
void pw_gpu_cff(const double *din, double *zout, const int *npts) {
  // Check inputs.
  assert(omp_get_num_threads() == 1);
  const int nrpts = npts[0] * npts[1] * npts[2];
  if (nrpts == 0) {
    return; // Nothing to do.
  }

  // Allocate device memory.
  offload_activate_chosen_device();
  const size_t buffer_size = 2 * sizeof(double) * nrpts;
  if (!pw_gpu_ensure_buffer_c(buffer_size, 0, false)) {
    host_real_fft(din, zout, npts, -1, 2);
    return;
  }

  // Upload REAL input and convert to COMPLEX on device.
  offloadMemcpyAsyncHtoD(buffer_dev_1, din, buffer_size / 2, stream);
  pw_gpu_launch_real_to_complex(buffer_dev_1, buffer_dev_2, nrpts, stream);

  // Run FFT on the device.
  // NOTE: Could use 2D-FFT, but CUDA does them C-shaped which is not optimal.
  fft_1d(OFFLOAD_FFT_FORWARD, npts[2], npts[0] * npts[1], buffer_dev_2,
         buffer_dev_1);
  fft_1d(OFFLOAD_FFT_FORWARD, npts[1], npts[0] * npts[2], buffer_dev_1,
         buffer_dev_2);

  // Download COMPLEX results to host.
  offloadMemcpyAsyncDtoH(zout, buffer_dev_2, buffer_size, stream);
  offloadStreamSynchronize(stream);
}

/*******************************************************************************
 * \brief   Performs a (double precision complex) 2D-FFT and a (double complex
 *          to double) shrink-down on the GPU.
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
void pw_gpu_ffc(const double *zin, double *dout, const int *npts) {
  // Check inputs.
  assert(omp_get_num_threads() == 1);
  const int nrpts = npts[0] * npts[1] * npts[2];
  if (nrpts == 0) {
    return; // Nothing to do.
  }

  // Allocate device memory.
  offload_activate_chosen_device();
  const size_t buffer_size = 2 * sizeof(double) * nrpts;
  if (!pw_gpu_ensure_buffer_c(buffer_size, 0, false)) {
    host_real_fft(zin, dout, npts, 1, 2);
    return;
  }

  // Upload COMPLEX input to device.
  offloadMemcpyAsyncHtoD(buffer_dev_1, zin, buffer_size, stream);

  // Run FFT on the device.
  // NOTE: Could use 2D-FFT, but CUDA does them C-shaped which is not optimal.
  fft_1d(OFFLOAD_FFT_INVERSE, npts[1], npts[0] * npts[2], buffer_dev_1,
         buffer_dev_2);
  fft_1d(OFFLOAD_FFT_INVERSE, npts[2], npts[0] * npts[1], buffer_dev_2,
         buffer_dev_1);
  pw_gpu_launch_complex_to_real(buffer_dev_1, buffer_dev_2, nrpts, stream);

  // Download REAL results to host.
  offloadMemcpyAsyncDtoH(dout, buffer_dev_2, buffer_size / 2, stream);
  offloadStreamSynchronize(stream);
}

/*******************************************************************************
 * \brief   Performs a (double to complex double) blow-up and a (double
 *          precision complex) 1D-FFT on the GPU.
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
void pw_gpu_cf(const double *din, double *zout, const int *npts) {
  // Check inputs.
  assert(omp_get_num_threads() == 1);
  const int nrpts = npts[0] * npts[1] * npts[2];
  if (nrpts == 0) {
    return; // Nothing to do.
  }

  // Allocate device memory.
  offload_activate_chosen_device();
  const size_t buffer_size = 2 * sizeof(double) * nrpts;
  if (!pw_gpu_ensure_buffer_c(buffer_size, 0, false)) {
    host_real_fft(din, zout, npts, -1, 1);
    return;
  }

  // Upload REAL input and convert to COMPLEX on device.
  offloadMemcpyAsyncHtoD(buffer_dev_1, din, buffer_size / 2, stream);
  pw_gpu_launch_real_to_complex(buffer_dev_1, buffer_dev_2, nrpts, stream);

  // Run FFT on the device.
  fft_1d(OFFLOAD_FFT_FORWARD, npts[2], npts[0] * npts[1], buffer_dev_2,
         buffer_dev_1);

  // Download COMPLEX results from device.
  offloadMemcpyAsyncDtoH(zout, buffer_dev_1, buffer_size, stream);
  offloadStreamSynchronize(stream);
}

/*******************************************************************************
 * \brief   Performs a (double precision complex) 1D-FFT and a (double complex
 *          to double) shrink-down on the GPU.
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
void pw_gpu_fc(const double *zin, double *dout, const int *npts) {
  // Check inputs.
  assert(omp_get_num_threads() == 1);
  const int nrpts = npts[0] * npts[1] * npts[2];
  if (nrpts == 0) {
    return; // Nothing to do.
  }

  // Allocate device memory.
  offload_activate_chosen_device();
  const size_t buffer_size = 2 * sizeof(double) * nrpts;
  if (!pw_gpu_ensure_buffer_c(buffer_size, 0, false)) {
    host_real_fft(zin, dout, npts, 1, 1);
    return;
  }

  // Upload COMPLEX input to device.
  offloadMemcpyAsyncHtoD(buffer_dev_1, zin, buffer_size, stream);

  // Run FFT on the device.
  fft_1d(OFFLOAD_FFT_INVERSE, npts[2], npts[0] * npts[1], buffer_dev_1,
         buffer_dev_2);

  // Convert COMPLEX results to REAL and download to host.
  pw_gpu_launch_complex_to_real(buffer_dev_2, buffer_dev_1, nrpts, stream);
  offloadMemcpyAsyncDtoH(dout, buffer_dev_1, buffer_size / 2, stream);
  offloadStreamSynchronize(stream);
}

/*******************************************************************************
 * \brief   Performs a (double precision complex) 1D-FFT on the GPU.
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
void pw_gpu_f(const double *zin, double *zout, const int dir, const int n,
              const int m) {
  // Check inputs.
  assert(omp_get_num_threads() == 1);
  const int nrpts = n * m;
  if (nrpts == 0) {
    return; // Nothing to do.
  }

  // Allocate device memory.
  offload_activate_chosen_device();
  const size_t buffer_size = 2 * sizeof(double) * nrpts;
  if (!pw_gpu_ensure_buffer_c(buffer_size, 0, false)) {
    host_fft_1d(dir > 0 ? -1 : 1, n, m, zin, zout);
    return;
  }

  // Upload COMPLEX input to device.
  offloadMemcpyAsyncHtoD(buffer_dev_1, zin, buffer_size, stream);

  // Run FFT on the device.
  if (dir > 0) {
    fft_1d(OFFLOAD_FFT_FORWARD, n, m, buffer_dev_1, buffer_dev_2);
  } else {
    fft_1d(OFFLOAD_FFT_INVERSE, n, m, buffer_dev_1, buffer_dev_2);
  }

  // Download COMPLEX results from device.
  offloadMemcpyAsyncDtoH(zout, buffer_dev_2, buffer_size, stream);
  offloadStreamSynchronize(stream);
}

/*******************************************************************************
 * \brief   Performs a (double precision complex) 1D-FFT, followed by a (double
 *          precision complex) gather, on the GPU.
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
void pw_gpu_fg(const double *zin, double *zout, const int *ghatmap,
               const int *npts, const int mmax, const int ngpts,
               const double scale) {
  // Check inputs.
  assert(omp_get_num_threads() == 1);
  const int nrpts = npts[0] * mmax;
  assert(ngpts <= nrpts);
  if (nrpts == 0 || ngpts == 0) {
    return; // Nothing to do.
  }

  // Allocate device memory.
  offload_activate_chosen_device();
  const size_t buffer_size = 2 * sizeof(double) * nrpts;
  const size_t map_size = sizeof(int) * ngpts;
  if (!pw_gpu_ensure_buffer_c(buffer_size, map_size, false)) {
    host_fg(zin, zout, ghatmap, npts[0], mmax, ngpts, scale);
    return;
  }

  // Upload COMPLEX inputs to device.
  offloadMemcpyAsyncHtoD(buffer_dev_1, zin, buffer_size, stream);

  // Run FFT on the device.
  fft_1d(OFFLOAD_FFT_FORWARD, npts[0], mmax, buffer_dev_1, buffer_dev_2);

  // Upload map and run gather on the device.
  offloadMemcpyAsyncHtoD(ghatmap_dev, ghatmap, map_size, stream);
  pw_gpu_launch_gather(buffer_dev_1, buffer_dev_2, scale, ngpts, ghatmap_dev,
                       stream);

  // Download COMPLEX results from device.
  offloadMemcpyAsyncDtoH(zout, buffer_dev_1, 2 * sizeof(double) * ngpts,
                         stream);
  offloadStreamSynchronize(stream);
}

/*******************************************************************************
 * \brief   Performs a (double precision complex) scatter, followed by a
 *          (double precision complex) 1D-FFT, on the GPU.
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
void pw_gpu_sf(const double *zin, double *zout, const int *ghatmap,
               const int *npts, const int mmax, const int ngpts,
               const int nmaps, const double scale) {
  // Check inputs.
  assert(omp_get_num_threads() == 1);
  const int nrpts = npts[0] * mmax;
  assert(ngpts <= nrpts);
  if (nrpts == 0 || ngpts == 0) {
    return; // Nothing to do.
  }

  // Allocate device memory.
  offload_activate_chosen_device();
  const size_t buffer_size = 2 * sizeof(double) * nrpts;
  const size_t map_size = sizeof(int) * nmaps * ngpts;
  if (!pw_gpu_ensure_buffer_c(buffer_size, map_size, false)) {
    host_sf(zin, zout, ghatmap, npts[0], mmax, ngpts, nmaps, scale);
    return;
  }

  // Upload COMPLEX inputs to device.
  offloadMemcpyAsyncHtoD(buffer_dev_1, zin, 2 * sizeof(double) * ngpts, stream);

  // Upload map and run scatter on the device.
  offloadMemcpyAsyncHtoD(ghatmap_dev, ghatmap, map_size, stream);
  offloadMemsetAsync(buffer_dev_2, 0, buffer_size, stream);
  pw_gpu_launch_scatter(buffer_dev_2, buffer_dev_1, scale, ngpts, nmaps,
                        ghatmap_dev, stream);

  // Run FFT on the device.
  fft_1d(OFFLOAD_FFT_INVERSE, npts[0], mmax, buffer_dev_2, buffer_dev_1);

  // Download COMPLEX results from device.
  offloadMemcpyAsyncDtoH(zout, buffer_dev_1, buffer_size, stream);
  offloadStreamSynchronize(stream);
}

#endif // defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)

// EOF
