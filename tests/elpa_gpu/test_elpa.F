PROGRAM test_elpa
   USE cp_fm_elpa
   USE cp_fm_types
   USE cp_fm_struct
   USE elpa_constants
   USE kinds, ONLY: dp
   USE message_passing
   USE OMP_LIB, ONLY: omp_set_num_threads, omp_get_max_threads
   USE test_support
   USE test_elpa_threshold, ONLY: configured_threshold => elpa_gpu_min_n
   IMPLICIT NONE
   TYPE(cp_fm_struct_type), POINTER :: layout
   TYPE(cp_fm_type) :: matrix, vectors
   INTEGER :: i, mask, neig, calls_before, expected_flags(4), requested_kernel
   INTEGER :: threshold = 12288, setup_before
   CHARACTER(16) :: argument

   CALL get_command_argument(1, argument)
   READ (argument, *) mask
   CALL get_command_argument(2, argument)
   IF (LEN_TRIM(argument) > 0) READ (argument, *) threshold
   CALL verify(configured_threshold() == threshold, "Incorrect calibrated AUTO threshold (default 12288)")
   support_gpu = BTEST(mask, 0)
   support_amd = BTEST(mask, 1)
   support_omp = BTEST(mask, 2)
   CALL omp_set_num_threads(1)

   ! Empty finalization and a fresh initialization must both work.
   CALL initialize_elpa_library()
   CALL finalize_elpa_library()
   CALL initialize_elpa_library(one_stage=.FALSE., qr=.FALSE., should_print=.FALSE.)
   ALLOCATE (layout)
   ALLOCATE (layout%context, layout%ncol_locals(1))
   CALL new_environment(1)
   neig = 2
   requested_kernel = ELPA_2STAGE_REAL_GENERIC
   CALL set_elpa_kernel(requested_kernel)
   CALL solve(.TRUE.)
   CALL verify(ALL(last_flags == 0), "Explicit CPU kernel enabled a GPU")
   DO i = 1, 10
      CALL solve(.FALSE.)
   END DO

   ! Each setup input must invalidate the cache independently.
   layout%nrow_global = 6
   CALL solve(.TRUE.)
   layout%nrow_block = 4
   CALL solve(.TRUE.)
   layout%local_leading_dimension = 9
   CALL solve(.TRUE.)
   layout%ncol_local = 9
   CALL solve(.TRUE.)
   neig = 3
   CALL solve(.TRUE.)
   layout%context%mepos(1) = 1
   CALL solve(.TRUE.)
   layout%context%mepos(2) = 1
   CALL solve(.TRUE.)
   CALL omp_set_num_threads(2)
   CALL solve(.TRUE.)

   ! Explicit AMD selection works without HIP; an explicit CPU kernel stays on the CPU.
   requested_kernel = ELPA_2STAGE_REAL_AMD_GPU
   CALL set_elpa_kernel(requested_kernel)
   CALL solve(.TRUE.)
   CALL solve(.FALSE.)
   requested_kernel = ELPA_2STAGE_REAL_GENERIC
   CALL set_elpa_kernel(requested_kernel)
   CALL solve(.TRUE.)
   fail_gpu_setup = .TRUE.
   requested_kernel = ELPA_2STAGE_REAL_AMD_GPU
   CALL set_elpa_kernel(requested_kernel)
   setup_before = gpu_setups
   CALL solve(.TRUE.)
   CALL solve(.FALSE.)
   CALL verify(gpu_setups == setup_before + MERGE(1, 0, support_gpu .OR. support_amd), "GPU setup retry")
   CALL verify(last_kernel == ELPA_2STAGE_REAL_AVX512_BLOCK4, "Failed GPU setup did not select SIMD")
   fail_gpu_setup = .FALSE.
   layout%nrow_block = 3
   CALL solve(.TRUE.)
   CALL verify(last_kernel == ELPA_2STAGE_REAL_AVX512_BLOCK4, "Invalid block did not fall back")
   support_avx512 = .FALSE.
   CALL solve(.TRUE.)
   CALL verify(last_kernel == ELPA_2STAGE_REAL_AVX2_BLOCK2, "AVX2 fallback order")
   support_avx2 = .FALSE.
   CALL solve(.TRUE.)
   CALL verify(last_kernel == ELPA_2STAGE_REAL_GENERIC, "Generic fallback order")
   support_avx512 = .TRUE.
   support_avx2 = .TRUE.
   layout%nrow_block = 1
   CALL solve(.TRUE.)
   layout%nrow_block = 2048
   CALL solve(.TRUE.)
   CALL verify(gpu_setups == setup_before + MERGE(1, 0, support_gpu .OR. support_amd), "Invalid block used GPU")
   layout%nrow_block = 4
   CALL solve(.TRUE.)
   requested_kernel = ELPA_2STAGE_REAL_GENERIC
   CALL set_elpa_kernel(requested_kernel)
   CALL solve(.TRUE.)
   elpa_one_stage = .TRUE.
   CALL solve(.TRUE.)
   CALL solve(.FALSE.)
   elpa_one_stage = .FALSE.
   CALL solve(.TRUE.)

   ! QR is mutable: diagnostic solves must not affect the next SCF iteration.
   elpa_qr = .TRUE.
   CALL solve(.FALSE.)
   CALL verify(last_qr == 1, "QR was not enabled on the cached handle")
   elpa_print = .TRUE.
   DO i = 1, 2
      calls_before = solves
      CALL solve(.FALSE.)
      CALL verify(solves == calls_before + 2, "QR diagnostic solve missing")
      CALL verify(previous_qr == 1 .AND. last_qr == 0, "QR leaked from the diagnostic solve")
   END DO
   elpa_print = .FALSE.
   CALL solve(.FALSE.)
   CALL verify(last_qr == 1, "QR not restored after diagnostic solve")
   elpa_qr = .FALSE.
   CALL solve(.FALSE.)
   CALL verify(last_qr == 0, "Disabled QR remained enabled")

   ! A mismatch on another rank must force this rank to participate in setup.
   remote_rebuild = 1
   CALL solve(.TRUE.)
   remote_rebuild = 0
   CALL solve(.FALSE.)

   ! Drop the caller's communicator reference while the cache still owns it.
   CALL mp_para_env_release(layout%para_env)
   CALL verify(comm_refs(1) == 1, "Cache did not retain the parent communicator")
   CALL new_environment(2)
   CALL solve(.TRUE.)
   CALL verify(comm_refs(1) == 0, "Evicted communicator leaked")

   ! AUTO resolves by build backend; the mock CPU reports MACHINE_CPU_GENERIC.
   CALL set_elpa_kernel(ELPA_2STAGE_REAL_INVALID)
   requested_kernel = ELPA_2STAGE_REAL_DEFAULT
#if !defined(__NO_OFFLOAD_ELPA)
#if defined(__OFFLOAD_CUDA)
   requested_kernel = ELPA_2STAGE_REAL_NVIDIA_GPU
#endif
#if defined(__OFFLOAD_HIP)
   requested_kernel = ELPA_2STAGE_REAL_AMD_GPU
   IF (layout%nrow_global < threshold) requested_kernel = ELPA_2STAGE_REAL_AVX512_BLOCK4
#endif
#if defined(__OFFLOAD_OPENCL)
   requested_kernel = ELPA_2STAGE_REAL_INTEL_GPU_SYCL
#endif
#endif
   CALL solve(.TRUE.)
   CALL solve(.FALSE.)
   requested_kernel = ELPA_2STAGE_REAL_GENERIC
   CALL set_elpa_kernel(requested_kernel)
   CALL solve(.TRUE.)
   CALL verify(ALL(last_flags == 0), "CPU selection after AUTO retained GPU flags")

   ! Other GPU backends must retain their existing enablement.
   requested_kernel = ELPA_2STAGE_REAL_NVIDIA_GPU
   CALL set_elpa_kernel(requested_kernel)
   CALL solve(.TRUE.)
   requested_kernel = ELPA_2STAGE_REAL_INTEL_GPU_SYCL
   CALL set_elpa_kernel(requested_kernel)
   CALL solve(.TRUE.)
   CALL mp_para_env_release(layout%para_env)
   CALL verify(comm_refs(2) == 1, "Cached communicator released too early")
   CALL finalize_elpa_library()
   CALL verify(ALL(comm_refs == 0), "Communicator leaked at finalization")

   ! Reinitialize with the same handle number and dimensions after cleanup.
   CALL initialize_elpa_library(one_stage=.FALSE., qr=.FALSE., should_print=.FALSE.)
   CALL new_environment(2)
   CALL solve(.TRUE.)
   CALL solve(.FALSE.)
   CALL finalize_elpa_library()
   CALL verify(comm_refs(2) == 1, "Finalization released the caller's communicator")
   CALL mp_para_env_release(layout%para_env)
   CALL verify(allocations == releases .AND. setups == allocations, "Unbalanced handle lifecycle")
   CALL verify(ALL(comm_refs == 0) .AND. timers == 0, "Leaked communicator or timer")
   DEALLOCATE (layout%context, layout%ncol_locals)
   DEALLOCATE (layout)
   PRINT *, "PASS cache lifecycle, GPU probes and QR; support mask", mask

CONTAINS
   SUBROUTINE new_environment(id)
      INTEGER, INTENT(IN) :: id
      ALLOCATE (layout%para_env)
      layout%para_env%id = id
      comm_refs(id) = comm_refs(id) + 1
   END SUBROUTINE

   SUBROUTINE solve(expect_new)
      LOGICAL, INTENT(IN) :: expect_new
      REAL(dp), ALLOCATABLE :: eval(:)
      INTEGER :: j, old_allocations, old_setups, old_reductions, expected_kernel
      LOGICAL :: amd_requested

      old_allocations = allocations
      old_setups = setups
      old_reductions = reductions
      layout%ncol_locals = layout%ncol_local
      CALL cp_fm_create(matrix, layout)
      CALL cp_fm_create(vectors, layout)
      ALLOCATE (eval(neig))
      matrix%local_data = 0.0_dp
      DO j = 1, layout%nrow_global
         matrix%local_data(j, j) = REAL(j, dp)
      END DO
      CALL cp_fm_diag_elpa(matrix, vectors, eval)
      CALL verify(ALL(eval == [(REAL(j, dp), j=1, neig)]), "Incorrect returned eigenvalues")
      CALL verify(allocations == old_allocations + MERGE(1, 0, expect_new), "Incorrect cache reuse or invalidation")
      CALL verify(setups == old_setups + MERGE(1, 0, expect_new), "Unexpected setup count")
      CALL verify(allocations == releases + 1, "Cache must keep exactly one handle alive")
      CALL verify(reductions >= old_reductions + 1, "Missing collective cache decision")
      CALL verify(ALL(last_layout == [layout%nrow_global, neig, layout%local_leading_dimension, &
                                     layout%ncol_local, layout%nrow_block, layout%para_env%id, &
                                     layout%context%mepos]), "Stale ELPA layout")
      CALL verify(last_solver == MERGE(ELPA_SOLVER_1STAGE, ELPA_SOLVER_2STAGE, elpa_one_stage), "Stale solver")
      CALL verify(last_threads == MERGE(omp_get_max_threads(), 0, support_omp), "Stale OpenMP thread count")
      amd_requested = requested_kernel == ELPA_2STAGE_REAL_AMD_GPU
      expected_kernel = requested_kernel
      IF (amd_requested) THEN
         IF (fail_gpu_setup .OR. .NOT. (support_gpu .OR. support_amd) .OR. &
             IAND(layout%nrow_block, layout%nrow_block - 1) /= 0 .OR. &
             layout%nrow_block < 2 .OR. layout%nrow_block > 1024) THEN
            amd_requested = .FALSE.
            expected_kernel = ELPA_2STAGE_REAL_GENERIC
            IF (support_avx2) expected_kernel = ELPA_2STAGE_REAL_AVX2_BLOCK2
            IF (support_avx512) expected_kernel = ELPA_2STAGE_REAL_AVX512_BLOCK4
         END IF
      END IF
      expected_flags = [MERGE(1, 0, amd_requested .AND. support_gpu .AND. .NOT. support_amd), &
                        MERGE(1, 0, amd_requested .AND. support_amd), &
                        MERGE(1, 0, requested_kernel == ELPA_2STAGE_REAL_NVIDIA_GPU), &
                        MERGE(1, 0, requested_kernel == ELPA_2STAGE_REAL_INTEL_GPU_SYCL)]
      CALL verify(ALL(last_flags == expected_flags), "Incorrect GPU enablement or stale backend")
      IF (.NOT. elpa_one_stage) CALL verify(last_kernel == expected_kernel, "Incorrect resolved kernel")
      CALL cp_fm_release(matrix)
      CALL cp_fm_release(vectors)
   END SUBROUTINE
END PROGRAM
