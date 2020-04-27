#include <cudnn.h>
#include <stdlib.h>
#include <cassert>

#define checkCUDNN(expression)                                                 \
  {                                                                            \
    cudnnStatus_t status = (expression);                                       \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      std::cerr << "Error on line " << __LINE__ << ": "                        \
                << cudnnGetErrorString(status) << "\n";                        \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

namespace gpu {
template <typename T>
T *gpu_malloc(size_t size) {
  void *memory{nullptr};
  auto error = cudaMalloc(&memory, size * sizeof(T));
  assert(error == cudaSuccess);
  return static_cast<T*>(memory);
}

template <typename T> T read_gpu_mem(T *gpu_mem, size_t idx) {
  T val{};
  auto error = cudaMemcpy(&val, gpu_mem, sizeof(T), cudaMemcpyDeviceToHost);
  assert(error == cudaSuccess);
  return val;
}

template <typename T> void write_gpu_mem(T *gpu_mem, size_t idx, T val) {
  auto error = cudaMemcpy(gpu_mem, &val, sizeof(T), cudaMemcpyHostToDevice);
  assert(error == cudaSuccess);
}

template <typename T> __global__ void fill_kernel(T *begin, T *end, T fillVal) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = begin + index; i < end; i += stride)
    *i = fillVal;
}

template <typename T> void fill(T *begin, T *end, T fillVal) {
    size_t N = end - begin;
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    fill_kernel<<<numBlocks, blockSize>>>(begin, end, fillVal);
}

template <typename T, typename Callable> __global__ void transform_kernel(T *begin, T *end, Callable func) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (auto i = begin + index; i < end; i += stride)
    *i = func(*i);
}

template <typename T, typename Callable> void transform(T *begin, T *end, Callable func) {
    size_t N = end - begin;
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    transform_kernel<<<numBlocks, blockSize>>>(begin, end, func);
}

} // namespace gpu
