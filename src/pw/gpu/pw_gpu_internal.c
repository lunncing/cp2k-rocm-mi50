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
  if (e->map_dev != NULL || e->g_dev != NULL) {
    // Uploads are asynchronous, including entries reclaimed during preparation.
    offloadStreamSynchronize(stream);
  }
  device_free(e->map_dev, sizeof(int) * (size_t)e->ngpts * (size_t)e->nmaps);
  device_free(e->g_dev, 3 * sizeof(double) * (size_t)e->ngpts);
  memset(e, 0, sizeof(grid_cache_entry));
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
  current_gpu_allocated_bytes = peak_gpu_allocated_bytes = 0;
  offload_activate_chosen_device();

  memset(grid_cache, 0, sizeof(grid_cache_entry) * PW_GPU_GRID_CACHE_SIZE);
  grid_cache_clock = 0;
  opt_map_cache = env_flag("CP2K_PW_GPU_MAP_CACHE", true);
  opt_fused_xc = env_flag("CP2K_PW_GPU_FUSED_XC", true);

  offloadStreamCreate(&stream);
  offloadStreamCreate(&stream_copy);
  for (int i = 0; i < 2; i++) {
    offloadEventCreate(&event_compute_done[i]);
    offloadEventCreate(&event_copy_done[i]);
  }
  is_initialized = true;
}

/*******************************************************************************
 * \brief Returns the runtime switches (bit 0: map cache, bit 1: fused XC).
 ******************************************************************************/
int pw_gpu_get_options(void) {
  read_memory_environment();
  if (!is_initialized) {
    return 0;
  }
  return (opt_map_cache ? 1 : 0) + (opt_fused_xc ? 2 : 0);
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
  release_work_buffers();
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
