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

add_library(pargraph_graph STATIC
  src/graph/csr_builder.cpp
)
target_include_directories(pargraph_graph PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
  target_compile_options(pargraph_graph PRIVATE /W4 /permissive-)
else()
  target_compile_options(pargraph_graph PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(PARGRAPH_BUILD_TOOLS)
  add_executable(convert_graph tools/convert_graph.cpp)
  target_link_libraries(convert_graph PRIVATE pargraph_graph)
endif()

if(PARGRAPH_BUILD_TESTS)
  enable_testing()
  add_executable(test_graph_csr tests/test_graph_csr.cpp)
  target_link_libraries(test_graph_csr PRIVATE pargraph_graph)
  target_compile_definitions(test_graph_csr PRIVATE
    PARGRAPH_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  )
  add_test(NAME graph_csr COMMAND test_graph_csr)
  set_tests_properties(graph_csr PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()

message(STATUS "pargraph Phase 0: graph CSR + convert_graph")
message(STATUS "  CUDA=${PARGRAPH_ENABLE_CUDA} MPI=${PARGRAPH_ENABLE_MPI}")
