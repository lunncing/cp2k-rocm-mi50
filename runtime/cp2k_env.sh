#!/usr/bin/env bash

# ==============================================================================
# CP2K & ROCm Environment Configuration (AMD MI50 + RX 9070 XT Dual-GPU)
# ==============================================================================

export ROCM_PATH=/opt/rocm/core-10.1
export HIP_PATH="$ROCM_PATH"

export PATH="/home/nowr/cp2k-hip/build-mi50/bin:$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/opt/dbcsr-rocm101/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CP2K_DATA_DIR=/home/nowr/cp2k-hip/data

unset HSA_OVERRIDE_GFX_VERSION

# CP2K / OpenMPI: disable hwloc GL detection & prevent X11 authorization warnings
export HWLOC_COMPONENTS=-gl

# CPU Threading for Ryzen 9 9950X (16C/32T)
export OMP_NUM_THREADS="${OMP_NUM_THREADS-16}"
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=FALSE
export OPENBLAS_NUM_THREADS=1
export OMP_STACKSIZE=256M
export CP2K_PW_GPU_RES_RESERVE_MB="${CP2K_PW_GPU_RES_RESERVE_MB-1024}"
export CP2K_PW_GPU_MEM_SAMPLE="${CP2K_PW_GPU_MEM_SAMPLE-1}"

# ------------------------------------------------------------------------------
# GPU Identity and Selection (Using permanent UUIDs to prevent index drift)
# ------------------------------------------------------------------------------
# MI50 (PCI 0000:03:00.0, gfx906, 6.7 TFLOPS FP64, 16GB HBM2 @ 1024 GB/s)
export MI50_UUID="GPU-37ac3981732c7340"
# RX 9070 XT (PCI 0000:07:00.0, gfx1201, 0.9 TFLOPS FP64, 16GB GDDR6 @ 624 GB/s)
export RX9070XT_UUID="GPU-6284fb6bbe682160"

# Default CP2K to MI50 (First-Principles Calculation is FP64 bound)
export ROCR_VISIBLE_DEVICES="$MI50_UUID"

# Select both the permanent wrapper and its GPU; discard previous formal entries.
_cp2k_path() {
    local entry selected="$1" rest=""
    local -a entries
    IFS=: read -r -a entries <<< "$PATH"
    for entry in "${entries[@]}"; do
        case "$entry" in
            /home/nowr/cp2k-hip/build-mi50/bin|/home/nowr/cp2k-hip/build-9070xt/bin) ;;
            *) rest="${rest:+$rest:}$entry" ;;
        esac
    done
    export PATH="/home/nowr/cp2k-hip/$selected/bin${rest:+:$rest}"
    hash -r
}

# Convenience functions for switching target GPU
use_mi50() {
    _cp2k_path build-mi50
    export ROCR_VISIBLE_DEVICES="$MI50_UUID"
    echo "[CP2K-ENV] Target GPU switched to MI50 ($MI50_UUID)"
}

use_9070xt() {
    _cp2k_path build-9070xt
    export ROCR_VISIBLE_DEVICES="$RX9070XT_UUID"
    echo "[CP2K-ENV] Target GPU switched to RX 9070 XT ($RX9070XT_UUID)"
}

# ------------------------------------------------------------------------------
# Automatic MI50 Hardware Clock Locking (HBM2 @ 1000 MHz, Core @ 1700 MHz)
# ------------------------------------------------------------------------------
lock_mi50_high() {
    local DPM_NODE="/sys/bus/pci/devices/0000:03:00.0/power_dpm_force_performance_level"
    if [ -f "$DPM_NODE" ]; then
        local CUR_DPM
        CUR_DPM=$(cat "$DPM_NODE" 2>/dev/null || true)
        if [ "$CUR_DPM" != "high" ]; then
            echo "[CP2K-ENV] MI50 DPM state is '$CUR_DPM'. Locking to 'high' (1000 MHz HBM2)..."
            docker run --rm --privileged -v /sys:/sys sha256:eb883e666c1d9787ce3696c50c2256c45f6c4a5107dc6af66c2a781cda7f4fd4 \
                -c "echo high > /sys/bus/pci/devices/0000:03:00.0/power_dpm_force_performance_level" >/dev/null 2>&1 || true
            local NEW_DPM
            NEW_DPM=$(cat "$DPM_NODE" 2>/dev/null || true)
            echo "[CP2K-ENV] MI50 DPM state is now: $NEW_DPM"
        fi
    fi
}

# Automatically ensure MI50 is at high performance whenever this file is sourced
lock_mi50_high
