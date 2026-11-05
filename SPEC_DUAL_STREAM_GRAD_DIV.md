# SPECIFICATION: Double Buffering & Dual-Stream Overlap in XC Gradient/Divergence

## Target Repository
Working tree: `D:/hermes/worktrees/cp2k-integrated`

## Objective
Overlap PCIe data transfers (DtoH in gradient, HtoD in divergence) with GPU FFT computation to hide ~150-200 ms of PCIe latency per SCF step.

## Requirements

### 1. Stream & Event Management
- In `src/pw/gpu/pw_gpu_internal.c`:
  - Manage a second offload stream `stream_copy` and synchronization events `event_compute_done[2]` and `event_copy_done[2]`.
  - Initialize on first use / in initialization routines.
  - Safely destroy during cleanup.

### 2. Double Buffers in `pw_gpu_ensure_buffer_c`
- When memory permits (`check_and_reclaim_memory` approves additional buffers):
  - Support secondary buffers:
    `static double *buffer_dev_1_b = NULL;` (real, size = buffer_size / 2)
    `static double *buffer_dev_2_b = NULL;` (complex, size = buffer_size)
  - Track allocations against `current_gpu_allocated_bytes`.
  - If memory limit does not allow allocating double buffers, gracefully keep using single buffer (fallback to serial stream).

### 3. Asynchronous Pipeline in `pw_gpu_sfffc_grad3`
- Overlap direction $d+1$'s `memset`, `scatter_deriv`, and `fft_3d` on `stream_compute` with direction $d$'s `offloadMemcpyAsyncDtoH` on `stream_copy`:
  - Direction $d=0$:
    - Compute on `stream_compute` using buffer A.
    - Record event.
    - Queue DtoH on `stream_copy` waiting on event.
  - Direction $d=1$:
    - Compute on `stream_compute` using buffer B in parallel with DtoH of direction 0!
    - Record event.
    - Queue DtoH on `stream_copy` waiting on event.
  - Direction $d=2$:
    - Wait for buffer A copy to complete, compute direction 2 on buffer A in parallel with DtoH of direction 1!
    - Queue DtoH on `stream_copy`.
  - Synchronize both streams before exiting.

### 4. Asynchronous Pipeline in `pw_gpu_cfffg_div3`
- Overlap direction $d+1$'s `offloadMemcpyAsyncHtoD` on `stream_copy` with direction $d$'s `fft_3d` and `gather_deriv_acc` on `stream_compute`.

### 5. Constraints
- Bit-for-bit identical numerical results.
- Full respect for `max_gpu_memory_bytes`, `host_fallback_enabled`, and LRU grid cache eviction.
- Fortran line lengths <= 132 characters.
- Add mock/unit tests testing dual stream execution and single-buffer fallback under memory constraint.
