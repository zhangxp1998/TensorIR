cmake_minimum_required(VERSION 3.0)


project(TensorIR)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)
set(ENGINE_KIND CPU CACHE STRING "Use CPU or GPU")
set(CMAKE_CXX_FLAGS "-fno-rtti")

find_package(MKLDNN REQUIRED)
find_package(MPI REQUIRED)

add_library(CPU_Runtime tensor.h tensor_diff.h tensor_constants.h tensor.cpp mpi_helper.h mpi_helper.cpp)
target_link_libraries(CPU_Runtime
  PUBLIC
  MKLDNN::mkldnn
)
add_library(GPU_Runtime gpu_tensor.h gpu_tensor.cpp)
target_link_libraries(GPU_Runtime
	PUBLIC
	cudnn
)

add_executable(snippet snippet.cpp)

target_link_libraries(snippet
  PRIVATE
  CPU_Runtime
  PRIVATE
  MPI::MPI_CXX)

add_executable(snippet_gpu snippet.cu)
target_link_libraries(snippet_gpu
  PRIVATE
  GPU_Runtime
  PRIVATE
  MPI::MPI_CXX)
