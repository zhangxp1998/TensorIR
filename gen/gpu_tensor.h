#ifndef __GPU_TENSOR_H
#define __GPU_TENSOR_H
#include <cudnn.h>
#include <stdlib.h>
#include <cassert>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <cublas_v2.h>

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

// template <typename T> __global__ void fill_kernel(T *begin, T *end, T fillVal) {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;
//   for (auto i = begin + index; i < end; i += stride)
//     *i = fillVal;
// }

template <typename T> void fill(T *begin, T *end, T fillVal) {
    thrust::fill(thrust::device_ptr<T>(begin), thrust::device_ptr<T>(end), fillVal);
}


template <typename T, typename Callable> void transform(T *begin, T *end, T *dst, Callable func) {
    thrust::transform(thrust::device_ptr<T>(begin), thrust::device_ptr<T>(end), thrust::device_ptr<T>(dst), std::move(func));
}

template <typename T, typename Callable> void transform(T *begin, T *end, T* begin2, T *dst, Callable func) {
    using thrust::device_ptr;
    thrust::transform(device_ptr<T>(begin), device_ptr<T>(end), device_ptr<T>(begin2), device_ptr<T>(dst), std::move(func));
}

template <typename T>
T *memdup(const T *src, size_t size) {
    T* dst = gpu_malloc<T>(size);
    auto error = cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToDevice);
    assert(error == cudaSuccess);
    return dst;
}

cublasHandle_t createCublasHandle();

extern cublasHandle_t cublasHandle;

// Expects data in row major format
void sgemm(const char transA, const char transB, const float *a, const float *b,
           float *c, const size_t M, const size_t K, const size_t N,
           const float& alpha, const float& beta);


} // namespace gpu

#endif