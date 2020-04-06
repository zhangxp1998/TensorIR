cmake_minimum_required(VERSION 3.0)


project(TensorIR)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)
set(CMAKE_CXX_FLAGS "-fno-rtti")

find_package(MKLDNN REQUIRED)
find_package(MPI REQUIRED)

add_executable(snippet snippet.cpp tensor.h tensor_diff.h tensor_constants.h tensor.cpp mpi_helper.h)

target_link_libraries(snippet
  PRIVATE
  MKLDNN::mkldnn
  PRIVATE
  MPI::MPI_CXX)
