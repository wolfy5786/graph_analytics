cmake_minimum_required(VERSION 3.20)
project(pargraph LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PARGRAPH_ENABLE_CUDA "Build CUDA kernels" OFF)
option(PARGRAPH_ENABLE_MPI "Build MPI variants" OFF)
option(PARGRAPH_ENABLE_SIMD "Compile AVX2 code paths" ON)
option(PARGRAPH_BUILD_TESTS "Build unit tests" ON)
option(PARGRAPH_BUILD_TOOLS "Build convert_graph and generate" ON)

find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
  message(FATAL_ERROR "OpenMP required for omp BFS (install libgomp / MSVC OpenMP)")
endif()

add_library(pargraph_core STATIC
  src/graph/csr_builder.cpp
  src/bfs/serial.cpp
  src/bfs/omp.cpp
  src/bfs/omp_atomic.cpp
  src/bfs/omp_diropt.cpp
  src/cc/serial.cpp
  src/tc/serial.cpp
  src/tc/omp.cpp
  src/tc/omp_edge.cpp
)
target_include_directories(pargraph_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(pargraph_core PUBLIC OpenMP::OpenMP_CXX)

if(MSVC)
  target_compile_options(pargraph_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(pargraph_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(PARGRAPH_ENABLE_MPI)
  set(PARGRAPH_MPI_ROOT "C:/msys64/ucrt64" CACHE PATH
      "Prefix with mpi.h and libmsmpi (MSYS2 ucrt64 msmpi)")
  find_path(PARGRAPH_MPI_INCLUDE mpi.h HINTS "${PARGRAPH_MPI_ROOT}/include")
  find_library(PARGRAPH_MPI_LIB NAMES msmpi libmsmpi
    HINTS "${PARGRAPH_MPI_ROOT}/lib")
  if(NOT PARGRAPH_MPI_INCLUDE OR NOT PARGRAPH_MPI_LIB)
    message(FATAL_ERROR
      "PARGRAPH_ENABLE_MPI=ON but mpi.h / msmpi were not found under "
      "${PARGRAPH_MPI_ROOT}. Install mingw-w64-ucrt-x86_64-msmpi.")
  endif()
  add_library(pargraph_mpi INTERFACE)
  target_include_directories(pargraph_mpi SYSTEM INTERFACE "${PARGRAPH_MPI_INCLUDE}")
  target_link_libraries(pargraph_mpi INTERFACE "${PARGRAPH_MPI_LIB}")
  target_compile_definitions(pargraph_mpi INTERFACE PARGRAPH_HAS_MPI=1)
  target_sources(pargraph_core PRIVATE src/bfs/mpi_atomic.cpp)
  target_link_libraries(pargraph_core PUBLIC pargraph_mpi)
  find_program(PARGRAPH_MPIEXEC NAMES mpiexec
    HINTS
      "$ENV{MSMPI_BIN}"
      "C:/Program Files/Microsoft MPI/Bin"
      "C:/Program Files (x86)/Microsoft MPI/Bin")
endif()

# Unified CLI driver (serial + OpenMP BFS, serial CC, optional MPI BFS).
add_executable(pargraph src/main.cpp)
target_link_libraries(pargraph PRIVATE pargraph_core)
if(PARGRAPH_ENABLE_MPI)
  # The validation suite is also linked here. Windows code-integrity policy on
  # this machine blocks brand-new unsigned exe names, while pargraph.exe is
  # already allowed to launch. `pargraph --mpi-self-test` is that entry point.
  target_sources(pargraph PRIVATE tests/test_bfs_mpi_atomic.cpp)
  target_compile_definitions(pargraph PRIVATE
    PARGRAPH_EMBED_MPI_TEST
    PARGRAPH_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  )
endif()

if(PARGRAPH_BUILD_TOOLS)
  add_executable(convert_graph tools/convert_graph.cpp)
  target_link_libraries(convert_graph PRIVATE pargraph_core)
endif()

if(PARGRAPH_BUILD_TESTS)
  enable_testing()

  add_executable(test_graph_csr tests/test_graph_csr.cpp)
  target_link_libraries(test_graph_csr PRIVATE pargraph_core)
  target_compile_definitions(test_graph_csr PRIVATE
    PARGRAPH_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  )
  add_test(NAME graph_csr COMMAND test_graph_csr)

  add_executable(test_bfs_serial tests/test_bfs_serial.cpp)
  target_link_libraries(test_bfs_serial PRIVATE pargraph_core)
  target_compile_definitions(test_bfs_serial PRIVATE
    PARGRAPH_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  )
  add_test(NAME bfs_serial COMMAND test_bfs_serial)

  add_executable(test_bfs_omp tests/test_bfs_omp.cpp)
  target_link_libraries(test_bfs_omp PRIVATE pargraph_core)
  target_compile_definitions(test_bfs_omp PRIVATE
    PARGRAPH_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  )
  add_test(NAME bfs_omp COMMAND test_bfs_omp)

  add_executable(test_bfs_omp_atomic tests/test_bfs_omp_atomic.cpp)
  target_link_libraries(test_bfs_omp_atomic PRIVATE pargraph_core)
  target_compile_definitions(test_bfs_omp_atomic PRIVATE
    PARGRAPH_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  )
  add_test(NAME bfs_omp_atomic COMMAND test_bfs_omp_atomic)

  add_executable(test_bfs_omp_diropt tests/test_bfs_omp_diropt.cpp)
  target_link_libraries(test_bfs_omp_diropt PRIVATE pargraph_core)
  target_compile_definitions(test_bfs_omp_diropt PRIVATE
    PARGRAPH_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  )
  add_test(NAME bfs_omp_diropt COMMAND test_bfs_omp_diropt)

  add_executable(bench_bfs_parallel tests/bench_bfs_parallel.cpp)
  target_link_libraries(bench_bfs_parallel PRIVATE pargraph_core)
  target_compile_definitions(bench_bfs_parallel PRIVATE
    PARGRAPH_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  )

  if(PARGRAPH_ENABLE_MPI AND PARGRAPH_MPIEXEC)
    add_executable(test_bfs_mpi_atomic tests/test_bfs_mpi_atomic.cpp)
    target_link_libraries(test_bfs_mpi_atomic PRIVATE pargraph_core)
    target_compile_definitions(test_bfs_mpi_atomic PRIVATE
      PARGRAPH_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )
    set(_mpi_path "C:/msys64/ucrt64/bin;C:/Program Files/Microsoft MPI/Bin;$ENV{PATH}")
    foreach(_np IN ITEMS 1 2 4 8)
      add_test(NAME bfs_mpi_atomic_np${_np}
        COMMAND "${PARGRAPH_MPIEXEC}" -n ${_np}
                $<TARGET_FILE:pargraph> --mpi-self-test)
      set_tests_properties(bfs_mpi_atomic_np${_np} PROPERTIES
        RUN_SERIAL TRUE
        ENVIRONMENT "PATH=${_mpi_path}"
        PROCESSORS ${_np})
    endforeach()
  endif()

  add_executable(test_cc_serial tests/test_cc_serial.cpp)
  target_link_libraries(test_cc_serial PRIVATE pargraph_core)
  target_compile_definitions(test_cc_serial PRIVATE
    PARGRAPH_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  )
  add_test(NAME cc_serial COMMAND test_cc_serial)

  add_executable(test_tc_serial tests/test_tc_serial.cpp)
  target_link_libraries(test_tc_serial PRIVATE pargraph_core)
  target_compile_definitions(test_tc_serial PRIVATE
    PARGRAPH_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  )
  add_test(NAME tc_serial COMMAND test_tc_serial)

  add_executable(test_tc_omp tests/test_tc_omp.cpp)
  target_link_libraries(test_tc_omp PRIVATE pargraph_core)
  target_compile_definitions(test_tc_omp PRIVATE
    PARGRAPH_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  )
  add_test(NAME tc_omp COMMAND test_tc_omp)

  add_executable(test_tc_omp_edge tests/test_tc_omp_edge.cpp)
  target_link_libraries(test_tc_omp_edge PRIVATE pargraph_core)
  target_compile_definitions(test_tc_omp_edge PRIVATE
    PARGRAPH_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  )
  add_test(NAME tc_omp_edge COMMAND test_tc_omp_edge)
endif()

message(STATUS "pargraph: serial+omp(+atomic,+diropt) BFS, serial CC, serial+omp(+edge) TC, optional MPI atomic BFS")
message(STATUS "  CUDA=${PARGRAPH_ENABLE_CUDA} MPI=${PARGRAPH_ENABLE_MPI} OpenMP=ON")
