/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef PW_GRID_PASSTHROUGH_H
#define PW_GRID_PASSTHROUGH_H

#include <stddef.h>

/* Alias-only adaptation of the verified stage3-spikes/grid_passthrough.c
 * bridge. These are borrowed views, never ABI casts of offload_buffer.
 * Layout/device identity and producer completion are checked by the callers;
 * the owner must remain alive until the synchronous consumer returns. */
typedef struct {
  void *dev_ptr;
  size_t nbytes;
} grid_buffer_descriptor;

static inline int grid_pw_bridge(const grid_buffer_descriptor *source,
                                 size_t bytes, grid_buffer_descriptor *out) {
  if (source == NULL || out == NULL || (bytes && source->dev_ptr == NULL) ||
      bytes > source->nbytes) return 0;
  out->dev_ptr = source->dev_ptr;
  out->nbytes = bytes;
  return 1;
}

#endif
