/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef OFFLOAD_BUFFER_H
#define OFFLOAD_BUFFER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
 * \brief Internal representation of a buffer.
 * \author Ole Schuett
 ******************************************************************************/
typedef struct {
  size_t size;
  double *host_buffer;
  double *device_buffer;
  /* Grid/PW pass-through state. device_buffer always retains ownership.
   * grid_device_view is a borrowed, read-only potential, valid only until
   * grid_integrate_task_list returns. Producers synchronize before publishing. */
  int grid_defer_host, grid_device_valid;
  size_t grid_bytes;
  double *grid_device_view;
} offload_buffer;

void offload_grid_prepare(offload_buffer *buffer, int defer_host);
double *offload_grid_device(const offload_buffer *buffer, size_t bytes);
int offload_grid_bind(offload_buffer *buffer, double *device, size_t bytes);
void offload_grid_host(offload_buffer *buffer);

/*******************************************************************************
 * \brief Allocate a buffer (NULL or valid) with the given number of elements.
 * \author Ole Schuett
 ******************************************************************************/
void offload_create_buffer(const int length, offload_buffer **buffer);

/*******************************************************************************
 * \brief Deallocate given buffer.
 * \author Ole Schuett
 ******************************************************************************/
void offload_free_buffer(offload_buffer *buffer);

/*******************************************************************************
 * \brief Return a pointer to the host buffer (Fortran API).
 * \author Ole Schuett
 ******************************************************************************/
double *offload_get_buffer_host_pointer(offload_buffer *buffer);

/*******************************************************************************
 * \brief Return a pointer to the device buffer (Fortran API).
 * \author Samal Bibek
 ******************************************************************************/
double *offload_get_buffer_device_pointer(offload_buffer *buffer);

/*******************************************************************************
 * \brief Copy data from host to device (Fortran API).
 * \author Samal Bibek
 ******************************************************************************/
void offload_buffer_h2d(offload_buffer *buffer, const int length);

/*******************************************************************************
 * \brief Copy data from device to host (Fortran API).
 * \author Samal Bibek
 ******************************************************************************/
void offload_buffer_d2h(offload_buffer *buffer, const int length);

#ifdef __cplusplus
}
#endif
#endif

// EOF
