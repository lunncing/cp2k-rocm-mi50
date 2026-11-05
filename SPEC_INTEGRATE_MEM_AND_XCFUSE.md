# SPECIFICATION: Integrate GPU Memory Limit / Host Fallback into CP2K XC-Fusion Tree

## Scope & Directives
Keep changes strictly restricted to: `src/input_cp2k_global.F`, `src/pw/pw_gpu.F`, `src/pw/gpu/pw_gpu_internal.c` (and any necessary header in `src/pw/gpu/`).
Do NOT attempt to add or modify other CP2K files (like `src/environment.F`).
Expose `pw_gpu_set_memory_limit` and `pw_gpu_set_host_fallback` in `pw_gpu.F`.
Crucially, environment variable detection (`CP2K_GPU_MEM_LIMIT_MB`, `CP2K_GPU_MEM_LIMIT_MB_RANK<N>`, `CP2K_GPU_HOST_FALLBACK`) must be performed automatically inside `pw_gpu_internal.c` on first call/initialization, so it works seamlessly even if Fortran input keywords are not explicitly wired at startup.

## Objectives
1. Implement configurable GPU memory budget and host fallback in `src/input_cp2k_global.F`, `src/pw/pw_gpu.F`, and `src/pw/gpu/pw_gpu_internal.c`.
2. Support rank-specific overrides via `CP2K_GPU_MEM_LIMIT_MB_RANK<N>` and global override `CP2K_GPU_MEM_LIMIT_MB`.
3. Support input section keywords `MAX_GPU_MEMORY_MB` and `HOST_FALLBACK` under `&GLOBAL &GRID`.
4. Integrate memory accounting into Claude's buffer management (`buffer_dev_1`, `buffer_dev_2`, `gbuf_dev`) and `grid_cache`.
5. When allocations would exceed the budget, evict LRU entries from `grid_cache`. If still exceeding and `HOST_FALLBACK` is enabled, gracefully fall back to host processing without crashing.
6. Preserve 100% of Claude's XC fusion (`pw_gpu_sfffc_grad3`, `pw_gpu_cfffg_div3`, `pw_gpu_filter_r3d`, Poisson dead code bypass, and multi-grid caching).
7. Fortran lines must NOT exceed 132 characters.
8. Add mock/unit tests in `tests/pw_gpu_integrated/` verifying memory limit enforcement, rank-specific parsing, LRU cache eviction under memory pressure, and fallback behavior.

## Detailed Requirements

### 1. `src/input_cp2k_global.F`
- In `create_global_section`:
  - Keyword `MAX_GPU_MEMORY_MB`: type INTEGER, default 0 (0 = unconstrained). Description: "Maximum GPU memory budget in MB per rank for plane-wave calculations".
  - Keyword `HOST_FALLBACK`: type LOGICAL, default .TRUE.. Description: "Allow plane-wave GPU routines to fall back to host memory/CPU when GPU memory limit is exceeded".
- In the routine that processes the `&GLOBAL` section (e.g. `read_global_section` or `pw_grid_setup`), call:
  - `CALL pw_gpu_set_memory_limit(max_gpu_memory_mb)`
  - `CALL pw_gpu_set_host_fallback(host_fallback)`

### 2. `src/pw/pw_gpu.F`
- Declare C bindings:
  - `pw_gpu_set_memory_limit_c(limit_mb)`
  - `pw_gpu_set_host_fallback_c(enable)`
  - `pw_gpu_get_allocated_bytes_c(allocated_bytes, peak_bytes)`
- Fortran wrapper subroutines:
  - `SUBROUTINE pw_gpu_set_memory_limit(limit_mb)`
  - `SUBROUTINE pw_gpu_set_host_fallback(enable)`

### 3. `src/pw/gpu/pw_gpu_internal.c`
- Memory state variables:
  - `static size_t max_gpu_memory_bytes = 0;`
  - `static int host_fallback_enabled = 1;`
  - `static size_t current_gpu_allocated_bytes = 0;`
  - `static size_t peak_gpu_allocated_bytes = 0;`
- Detection logic:
  - At module initialization or first query, check environment variables:
    - Determine rank: inspect `OMPI_COMM_WORLD_RANK`, `PMI_RANK`, `SLURM_PROCID`, `MPI_LOCALRANKID`. If none found, rank = 0.
    - Check `CP2K_GPU_MEM_LIMIT_MB_RANK<rank>`.
    - If not set, check `CP2K_GPU_MEM_LIMIT_MB`.
    - If found and > 0, set `max_gpu_memory_bytes = (size_t)mb * 1024ULL * 1024ULL;`
    - Check `CP2K_GPU_HOST_FALLBACK`: if set to "0" or "FALSE", set `host_fallback_enabled = 0`.
- Memory checking and eviction:
  - `static int check_and_reclaim_memory(size_t required_bytes)`:
    - If `max_gpu_memory_bytes == 0` (unlimited), return 1.
    - If `current_gpu_allocated_bytes + required_bytes <= max_gpu_memory_bytes`, return 1.
    - If exceeding, attempt to evict entries from `grid_cache` (freeing `map_dev` and `g_dev`, updating `current_gpu_allocated_bytes`).
    - If after eviction `current_gpu_allocated_bytes + required_bytes <= max_gpu_memory_bytes`, return 1.
    - If still exceeding: return 0 (budget exceeded).
- Buffer allocation integration:
  - In `pw_gpu_ensure_buffer_c`, `pw_gpu_ensure_gbuf_c`, and `grid_cache_lookup_or_upload`:
    - Call `check_and_reclaim_memory(required_size)`.
    - If it returns 0:
      - If `host_fallback_enabled`: print a one-time notice `[PW_GPU] Memory limit reached (%zu MB), falling back to host.` and return appropriate fallback status (e.g. NULL pointer or fallback return code).
- Memory tracking:
  - Track every `offloadMalloc` and `offloadFree` against `current_gpu_allocated_bytes` and update `peak_gpu_allocated_bytes`.

### 4. Tests
- Add C-based mock test in `tests/pw_gpu_integrated/` testing:
  - Setting limit and checking allocation rejection / fallback.
  - Environment variable parsing for rank 0 vs rank 1.
  - LRU cache eviction under memory pressure.
