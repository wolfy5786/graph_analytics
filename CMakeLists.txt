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

add_library(pargraph_core STATIC
  src/graph/csr_builder.cpp
  src/bfs/serial.cpp
)
target_include_directories(pargraph_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
  target_compile_options(pargraph_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(pargraph_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Unified CLI driver (Phase 1: serial BFS).
add_executable(pargraph src/main.cpp)
target_link_libraries(pargraph PRIVATE pargraph_core)

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
endif()

message(STATUS "pargraph: serial BFS + CSR tooling")
message(STATUS "  CUDA=${PARGRAPH_ENABLE_CUDA} MPI=${PARGRAPH_ENABLE_MPI}")
