! Minimal dependencies for compiling the complete production cp_fm_elpa module.
MODULE test_support
   IMPLICIT NONE
   INTEGER :: allocations = 0, releases = 0, setups = 0, solves = 0, timers = 0
   INTEGER :: remote_rebuild = 0, reductions = 0, comm_refs(16) = 0
   INTEGER :: last_qr = -1, previous_qr = -1, last_flags(4) = 0
   INTEGER :: last_layout(8) = 0, last_solver = 0, last_threads = 0
   LOGICAL :: support_gpu = .TRUE., support_amd = .TRUE., support_omp = .TRUE., initialized = .FALSE.
CONTAINS
   SUBROUTINE verify(condition, message)
      LOGICAL, INTENT(IN) :: condition
      CHARACTER(*), INTENT(IN) :: message
      IF (.NOT. condition) THEN
         PRINT *, message
         ERROR STOP 1
      END IF
   END SUBROUTINE
   SUBROUTINE abort_test(message)
      CHARACTER(*), INTENT(IN) :: message
      PRINT *, message
      ERROR STOP 2
   END SUBROUTINE
   SUBROUTINE warn_test(condition, message)
      LOGICAL, INTENT(IN) :: condition
      CHARACTER(*), INTENT(IN) :: message
      CALL verify(.NOT. condition, message)
   END SUBROUTINE
   SUBROUTINE timeset(name, handle)
      CHARACTER(*), INTENT(IN) :: name
      INTEGER, INTENT(OUT) :: handle
      timers = timers + 1
      handle = timers
   END SUBROUTINE
   SUBROUTINE timestop(handle)
      INTEGER, INTENT(IN) :: handle
      timers = timers - 1
   END SUBROUTINE
END MODULE

MODULE kinds
   USE ISO_FORTRAN_ENV, ONLY: real64
   IMPLICIT NONE
   INTEGER, PARAMETER :: dp = real64, default_string_length = 256
END MODULE

MODULE machine
   IMPLICIT NONE
   INTEGER, PARAMETER :: MACHINE_CPU_GENERIC = 0, MACHINE_X86_SSE4 = 1, MACHINE_X86_AVX = 2
   INTEGER, PARAMETER :: MACHINE_X86_AVX2 = 3, MACHINE_X86 = 4
CONTAINS
   INTEGER FUNCTION m_cpuid()
      m_cpuid = MACHINE_CPU_GENERIC
   END FUNCTION
END MODULE

MODULE message_passing
   USE test_support
   IMPLICIT NONE
   TYPE mp_comm_type
      INTEGER :: id = 1
   CONTAINS
      PROCEDURE :: assign_comm
      GENERIC :: ASSIGNMENT(=) => assign_comm
      PROCEDURE :: get_handle
      PROCEDURE :: bcast
      PROCEDURE :: max => comm_max
   END TYPE
   TYPE, EXTENDS(mp_comm_type) :: mp_para_env_type
      INTEGER :: ref_count = 1
   CONTAINS
      PROCEDURE :: retain
   END TYPE
CONTAINS
   SUBROUTINE assign_comm(this, other)
      CLASS(mp_comm_type), INTENT(OUT) :: this
      CLASS(mp_comm_type), INTENT(IN) :: other
      this%id = other%id
   END SUBROUTINE
   INTEGER FUNCTION get_handle(this)
      CLASS(mp_comm_type), INTENT(IN) :: this
      get_handle = this%id
   END FUNCTION
   SUBROUTINE bcast(this, value)
      CLASS(mp_comm_type), INTENT(IN) :: this
      LOGICAL, INTENT(INOUT) :: value
   END SUBROUTINE
   SUBROUTINE comm_max(this, value)
      CLASS(mp_comm_type), INTENT(IN) :: this
      INTEGER, INTENT(INOUT) :: value
      reductions = reductions + 1
      value = MAX(value, remote_rebuild)
   END SUBROUTINE
   SUBROUTINE retain(this)
      CLASS(mp_para_env_type), INTENT(INOUT) :: this
      this%ref_count = this%ref_count + 1
      comm_refs(this%id) = comm_refs(this%id) + 1
   END SUBROUTINE
   SUBROUTINE mp_para_env_release(env)
      TYPE(mp_para_env_type), POINTER :: env
      IF (ASSOCIATED(env)) THEN
         env%ref_count = env%ref_count - 1
         comm_refs(env%id) = comm_refs(env%id) - 1
         CALL verify(comm_refs(env%id) >= 0, "Communicator released too often")
         IF (env%ref_count == 0) DEALLOCATE (env)
      END IF
      NULLIFY (env)
   END SUBROUTINE
END MODULE

MODULE cp_blacs_env
   IMPLICIT NONE
   TYPE cp_blacs_env_type
      INTEGER :: mepos(2) = 0
   END TYPE
END MODULE

MODULE cp_fm_struct
   USE message_passing, ONLY: mp_para_env_type
   USE cp_blacs_env, ONLY: cp_blacs_env_type
   IMPLICIT NONE
   TYPE cp_fm_struct_type
      INTEGER :: nrow_global = 4, nrow_block = 2, local_leading_dimension = 8, ncol_local = 8
      INTEGER, POINTER :: ncol_locals(:) => NULL()
      TYPE(mp_para_env_type), POINTER :: para_env => NULL()
      TYPE(cp_blacs_env_type), POINTER :: context => NULL()
   END TYPE
CONTAINS
   SUBROUTINE cp_fm_struct_get(struct, local_leading_dimension, ncol_local, nrow_block, ncol_locals)
      TYPE(cp_fm_struct_type), INTENT(IN) :: struct
      INTEGER, INTENT(OUT) :: local_leading_dimension, ncol_local, nrow_block
      INTEGER, POINTER :: ncol_locals(:)
      local_leading_dimension = struct%local_leading_dimension
      ncol_local = struct%ncol_local
      nrow_block = struct%nrow_block
      ncol_locals => struct%ncol_locals
   END SUBROUTINE
END MODULE

MODULE cp_fm_types
   USE kinds, ONLY: dp
   USE cp_fm_struct, ONLY: cp_fm_struct_type
   IMPLICIT NONE
   TYPE cp_fm_type
      TYPE(cp_fm_struct_type), POINTER :: matrix_struct => NULL()
      REAL(dp), POINTER :: local_data(:, :) => NULL()
   END TYPE
CONTAINS
   SUBROUTINE cp_fm_create(matrix, matrix_struct)
      TYPE(cp_fm_type), INTENT(OUT) :: matrix
      TYPE(cp_fm_struct_type), POINTER :: matrix_struct
      matrix%matrix_struct => matrix_struct
      ALLOCATE (matrix%local_data(matrix_struct%local_leading_dimension, matrix_struct%ncol_local))
   END SUBROUTINE
   SUBROUTINE cp_fm_release(matrix)
      TYPE(cp_fm_type), INTENT(INOUT) :: matrix
      DEALLOCATE (matrix%local_data)
      NULLIFY (matrix%matrix_struct)
   END SUBROUTINE
   SUBROUTINE cp_fm_to_fm(source, destination)
      TYPE(cp_fm_type), INTENT(IN) :: source, destination
      destination%local_data = source%local_data
   END SUBROUTINE
   SUBROUTINE cp_fm_write_info(matrix, unit)
      TYPE(cp_fm_type), INTENT(IN) :: matrix
      INTEGER, INTENT(IN) :: unit
   END SUBROUTINE
END MODULE

MODULE cp_fm_basic_linalg
   USE cp_fm_types, ONLY: cp_fm_type, cp_fm_to_fm
   IMPLICIT NONE
CONTAINS
   SUBROUTINE cp_fm_uplo_to_full(matrix, eigenvectors)
      TYPE(cp_fm_type), INTENT(IN) :: matrix, eigenvectors
      CALL cp_fm_to_fm(matrix, eigenvectors)
   END SUBROUTINE
END MODULE

MODULE cp_fm_diag_utils
   USE kinds, ONLY: dp
   USE cp_fm_types, ONLY: cp_fm_type
   IMPLICIT NONE
   TYPE cp_fm_redistribute_info
   CONTAINS
      PROCEDURE :: write => write_info
   END TYPE
CONTAINS
   SUBROUTINE write_info(this, unit)
      CLASS(cp_fm_redistribute_info), INTENT(IN) :: this
      INTEGER, INTENT(IN) :: unit
   END SUBROUTINE
   SUBROUTINE cp_fm_redistribute_start(matrix, eigenvectors, matrix_new, eigenvectors_new, caller_is_elpa, redist_info)
      TYPE(cp_fm_type), INTENT(IN) :: matrix, eigenvectors
      TYPE(cp_fm_type), INTENT(OUT) :: matrix_new, eigenvectors_new
      LOGICAL, INTENT(IN) :: caller_is_elpa
      TYPE(cp_fm_redistribute_info), INTENT(OUT) :: redist_info
      matrix_new = matrix
      eigenvectors_new = eigenvectors
   END SUBROUTINE
   SUBROUTINE cp_fm_redistribute_end(matrix, eigenvectors, eigenvalues, matrix_new, eigenvectors_new)
      TYPE(cp_fm_type), INTENT(IN) :: matrix, eigenvectors
      TYPE(cp_fm_type), INTENT(INOUT) :: matrix_new, eigenvectors_new
      REAL(dp), INTENT(INOUT) :: eigenvalues(:)
   END SUBROUTINE
END MODULE

MODULE cp_log_handling
   IMPLICIT NONE
   TYPE cp_logger_type
      INTEGER :: unused = 0
   END TYPE
   TYPE(cp_logger_type), TARGET :: default_logger
CONTAINS
   FUNCTION cp_get_default_logger() RESULT(logger)
      TYPE(cp_logger_type), POINTER :: logger
      logger => default_logger
   END FUNCTION
   INTEGER FUNCTION cp_logger_get_default_io_unit(logger)
      TYPE(cp_logger_type), POINTER :: logger
      cp_logger_get_default_io_unit = -1
   END FUNCTION
   FUNCTION cp_to_string(value) RESULT(string)
      INTEGER, INTENT(IN) :: value
      CHARACTER(32) :: string
      WRITE (string, '(I0)') value
   END FUNCTION
END MODULE

MODULE elpa
   USE elpa_constants
   USE kinds, ONLY: dp
   USE test_support
   IMPLICIT NONE
   ! The real elpa_t is abstract: a non-polymorphic TYPE(elpa_t) cache must not compile.
   TYPE, ABSTRACT :: elpa_t
      INTEGER :: layout(8) = 0, flags(4) = 0, solver = 0, kernel = 0, qr = 0, threads = 0
      LOGICAL :: ready = .FALSE.
   CONTAINS
      PROCEDURE :: set
      PROCEDURE :: can_set
      PROCEDURE :: setup
      PROCEDURE :: eigenvectors
   END TYPE
   TYPE, EXTENDS(elpa_t) :: mock_elpa_t
   END TYPE
CONTAINS
   INTEGER FUNCTION elpa_init(version)
      INTEGER, INTENT(IN) :: version
      CALL verify(.NOT. initialized, "Double initialization")
      initialized = .TRUE.
      elpa_init = ELPA_OK
   END FUNCTION
   SUBROUTINE elpa_uninit()
      CALL verify(initialized, "ELPA is not initialized")
      CALL verify(allocations == releases, "ELPA finalized with a live handle")
      initialized = .FALSE.
   END SUBROUTINE
   FUNCTION elpa_allocate(error) RESULT(obj)
      INTEGER, OPTIONAL, INTENT(OUT) :: error
      CLASS(elpa_t), POINTER :: obj
      CALL verify(initialized, "Allocation before ELPA initialization")
      CALL verify(allocations == releases, "Old cache was not released before replacement")
      ALLOCATE (mock_elpa_t :: obj)
      allocations = allocations + 1
      IF (PRESENT(error)) error = ELPA_OK
   END FUNCTION
   SUBROUTINE elpa_deallocate(obj, error)
      CLASS(elpa_t), POINTER :: obj
      INTEGER, INTENT(OUT) :: error
      CALL verify(initialized, "Deallocation after ELPA finalization")
      CALL verify(comm_refs(obj%layout(6)) > 0, "Parent communicator released before ELPA handle")
      DEALLOCATE (obj)
      releases = releases + 1
      error = ELPA_OK
   END SUBROUTINE
   INTEGER FUNCTION can_set(this, key, value)
      CLASS(elpa_t), INTENT(IN) :: this
      CHARACTER(*), INTENT(IN) :: key
      INTEGER, INTENT(IN) :: value
      LOGICAL :: supported
      SELECT CASE (key)
      CASE ("gpu")
         supported = support_gpu
      CASE ("amd-gpu")
         supported = support_amd
      CASE ("omp_threads")
         supported = support_omp
      CASE DEFAULT
         supported = .FALSE.
      END SELECT
      can_set = MERGE(ELPA_OK, -1, supported)
   END FUNCTION
   SUBROUTINE set(this, key, value, error)
      CLASS(elpa_t), INTENT(INOUT) :: this
      CHARACTER(*), INTENT(IN) :: key
      INTEGER, INTENT(IN) :: value
      INTEGER, INTENT(OUT) :: error
      INTEGER :: index
      index = 0
      SELECT CASE (key)
      CASE ("na")
         index = 1
      CASE ("nev")
         index = 2
      CASE ("local_nrows")
         index = 3
      CASE ("local_ncols")
         index = 4
      CASE ("nblk")
         index = 5
      CASE ("mpi_comm_parent")
         index = 6
      CASE ("process_row")
         index = 7
      CASE ("process_col")
         index = 8
      CASE ("solver")
         this%solver = value
      CASE ("gpu", "amd-gpu", "omp_threads")
         error = this%can_set(key, value)
         IF (error /= ELPA_OK) RETURN
         IF (key == "gpu") this%flags(1) = value
         IF (key == "amd-gpu") this%flags(2) = value
         IF (key == "omp_threads") this%threads = value
      CASE ("nvidia-gpu")
         this%flags(3) = value
      CASE ("intel-gpu")
         this%flags(4) = value
      CASE ("real_kernel")
         CALL verify(this%solver == ELPA_SOLVER_2STAGE, "Kernel set on one-stage solver")
         IF (value == ELPA_2STAGE_REAL_AMD_GPU) THEN
            CALL verify((this%flags(1) == 1) .EQV. support_gpu, "Generic GPU flag missing before kernel")
            CALL verify((this%flags(2) == 1) .EQV. support_amd, "AMD GPU flag missing before kernel")
         END IF
         this%kernel = value
      CASE ("qr")
         CALL verify(this%solver == ELPA_SOLVER_2STAGE, "QR set on one-stage solver")
         this%qr = value
      CASE DEFAULT
         CALL abort_test("Unknown ELPA option: "//key)
      END SELECT
      IF (index > 0) THEN
         CALL verify(.NOT. this%ready, "Geometry changed after setup")
         this%layout(index) = value
      END IF
      error = ELPA_OK
   END SUBROUTINE
   INTEGER FUNCTION setup(this)
      CLASS(elpa_t), INTENT(INOUT) :: this
      CALL verify(.NOT. this%ready, "Repeated setup on a cached handle")
      CALL verify(ALL(this%layout(1:6) > 0), "Incomplete ELPA geometry")
      this%ready = .TRUE.
      setups = setups + 1
      setup = ELPA_OK
   END FUNCTION
   SUBROUTINE eigenvectors(this, matrix, eval, vectors, error)
      CLASS(elpa_t), INTENT(INOUT) :: this
      REAL(dp), INTENT(INOUT) :: matrix(:, :)
      REAL(dp), INTENT(OUT) :: eval(:), vectors(:, :)
      INTEGER, INTENT(OUT) :: error
      INTEGER :: i
      CALL verify(this%ready .AND. initialized, "Eigensolver called without setup")
      CALL verify(SIZE(eval) == this%layout(1), "Stale global size")
      CALL verify(ALL(SHAPE(matrix) == this%layout(3:4)), "Stale local dimensions")
      CALL verify(comm_refs(this%layout(6)) > 0, "Stale communicator")
      vectors = 0.0_dp
      DO i = 1, SIZE(eval)
         eval(i) = matrix(i, i)
         vectors(i, i) = 1.0_dp
      END DO
      matrix = -1.0_dp
      previous_qr = last_qr
      last_qr = this%qr
      last_flags = this%flags
      last_layout = this%layout
      last_solver = this%solver
      last_threads = this%threads
      solves = solves + 1
      error = ELPA_OK
   END SUBROUTINE
END MODULE
