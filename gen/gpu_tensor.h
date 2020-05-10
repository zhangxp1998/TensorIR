#ifndef __GPU_TENSOR_H
#define __GPU_TENSOR_H
#include <cudnn.h>
#include <stdlib.h>
#include <cassert>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include "tensor_constants.h"

#define checkCUDNN(expression)                          \
  {                                                     \
    cudnnStatus_t status = (expression);                \
    if (status != CUDNN_STATUS_SUCCESS)                 \
    {                                                   \
      std::cerr << "Error on line " << __FILE__ << ":"  \
                << __LINE__ << " "                      \
                << cudnnGetErrorString(status) << "\n"; \
      assert(status == CUDNN_STATUS_SUCCESS);           \
    }                                                   \
  }

namespace gpu
{
template <typename T>
T *gpu_malloc(size_t size)
{
  void *memory{nullptr};
  auto error = cudaMalloc(&memory, size * sizeof(T));
  assert(error == cudaSuccess);
  return static_cast<T *>(memory);
}

void gpu_free(void *p);

template <typename T>
T read_gpu_mem(T *gpu_mem, size_t idx)
{
  T val{};
  auto error = cudaMemcpy(&val, gpu_mem + idx, sizeof(T), cudaMemcpyDeviceToHost);
  assert(error == cudaSuccess);
  return val;
}

template <typename T>
void write_gpu_mem(T *gpu_mem, size_t idx, T val)
{
  auto error = cudaMemcpy(gpu_mem + idx, &val, sizeof(T), cudaMemcpyHostToDevice);
  assert(error == cudaSuccess);
}

void *memcpy(void *dest, const void *src, size_t n);

template <typename T>
void fill(T *begin, T *end, T fillVal)
{
  thrust::fill(thrust::device_ptr<T>(begin), thrust::device_ptr<T>(end), fillVal);
}

template <typename T, typename Callable>
void transform(T *begin, T *end, T *dst, Callable func)
{
  thrust::transform(thrust::device_ptr<T>(begin), thrust::device_ptr<T>(end), thrust::device_ptr<T>(dst), std::move(func));
}

template <typename T, typename Callable>
void transform(T *begin, T *end, T *begin2, T *dst, Callable func)
{
  using thrust::device_ptr;
  thrust::transform(device_ptr<T>(begin), device_ptr<T>(end), device_ptr<T>(begin2), device_ptr<T>(dst), std::move(func));
}

template <typename T>
T *memdup(const T *src, size_t size)
{
  T *dst = gpu_malloc<T>(size);
  auto error = cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToDevice);
  assert(error == cudaSuccess);
  return dst;
}

cublasHandle_t createCublasHandle();
cudnnHandle_t createCudnnHandle();

extern cublasHandle_t cublasHandle;
extern cudnnHandle_t cudnnHandle;

// Expects data in row major format
void sgemm(const char transA, const char transB, const float *a, const float *b,
           float *c, const size_t M, const size_t K, const size_t N,
           const float &alpha, const float &beta);

void matmul_backprop(const float *m1, const float *m2, const float *y,
                     float *d1, float *d2, const size_t M, const size_t K,
                     const size_t N);

template <typename T>
constexpr cudnnDataType_t get_cudnn_type() noexcept;

template <>
constexpr cudnnDataType_t get_cudnn_type<float>() noexcept
{
  return CUDNN_DATA_FLOAT;
}

template <>
constexpr cudnnDataType_t get_cudnn_type<double>() noexcept
{
  return CUDNN_DATA_DOUBLE;
}

template <>
constexpr cudnnDataType_t get_cudnn_type<int8_t>() noexcept
{
  return CUDNN_DATA_INT8;
}

template <>
constexpr cudnnDataType_t get_cudnn_type<uint8_t>() noexcept
{
  return CUDNN_DATA_UINT8;
}

template <>
constexpr cudnnDataType_t get_cudnn_type<int32_t>() noexcept
{
  return CUDNN_DATA_INT32;
}

template <typename T, cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW>
cudnnTensorDescriptor_t createTensor4dDescriptor(size_t N, size_t C, size_t H, size_t W)
{
  cudnnTensorDescriptor_t tensor_descriptor{};
  checkCUDNN(cudnnCreateTensorDescriptor(&tensor_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(tensor_descriptor,
                                        /*format=*/format,
                                        /*dataType=*/get_cudnn_type<T>(),
                                        /*batch_size=*/N,
                                        /*channels=*/C,
                                        /*image_height=*/H,
                                        /*image_width=*/W));
  return tensor_descriptor;
}

template <size_t N, size_t C, size_t H, size_t W, typename T, cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW>
cudnnTensorDescriptor_t getTensor4dDescriptor()
{
  static cudnnTensorDescriptor_t tensor_descriptor = createTensor4dDescriptor<T, format>(N, C, H, W);
  return tensor_descriptor;
}

template <typename T, cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW>
cudnnFilterDescriptor_t createTensor4dFilterDescriptor(size_t N, size_t C, size_t H, size_t W)
{
  cudnnFilterDescriptor_t kernel_descriptor{};
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                        get_cudnn_type<T>(),
                                        CUDNN_TENSOR_NCHW,
                                        N,
                                        C,
                                        H,
                                        W));
  return kernel_descriptor;
}

template <size_t N, size_t C, size_t H, size_t W, typename T, cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW>
cudnnFilterDescriptor_t getFilter4dDescriptor()
{
  static auto kernel_descriptor = createTensor4dFilterDescriptor<T, format>(N, C, H, W);
  return kernel_descriptor;
}
cudnnActivationDescriptor_t createActivationDescriptor(cudnnActivationMode_t activationMode, double coef);

template <typename T>
cudnnConvolutionDescriptor_t createConvolutionDescriptor(size_t padding, size_t stride)
{
  cudnnConvolutionDescriptor_t convolution_descriptor{};
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(
      cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                      /*pad_height=*/padding,
                                      /*pad_width=*/padding,
                                      /*vertical_stride=*/stride,
                                      /*horizontal_stride=*/stride,
                                      /*dilation_height=*/1,
                                      /*dilation_width=*/1,
                                      /*mode=*/CUDNN_CROSS_CORRELATION,
                                      /*computeType=*/get_cudnn_type<T>()));
  return convolution_descriptor;
}

template <size_t padding, size_t stride, typename T>
cudnnConvolutionDescriptor_t getConvolutionDescriptor()
{
  static auto descriptor = createConvolutionDescriptor<T>(padding, stride);
  return descriptor;
}

cudnnConvolutionFwdAlgo_t getConvolutionAlgo(cudnnHandle_t handle, cudnnTensorDescriptor_t input_descriptor, cudnnTensorDescriptor_t output_descriptor, cudnnFilterDescriptor_t kernel_descriptor, cudnnConvolutionDescriptor_t convolution_descriptor);

template <size_t N, size_t C, size_t H, size_t W, size_t OutChannels,
          size_t KernelSize, size_t padding, size_t stride, typename T>
void conv2d_forward(cudnnHandle_t handle,
                    const T *input, T *output,
                    const T *weights, const T *bias)
{

  static auto input_descriptor = getTensor4dDescriptor<N, C, H, W, T>();
  static auto output_descriptor = getTensor4dDescriptor<N, OutChannels, H, W, T>();
  static auto bias_descriptor = getTensor4dDescriptor<1, OutChannels, 1, 1, T>();
  static auto kernel_descriptor = getFilter4dDescriptor<OutChannels, C, KernelSize, KernelSize, T>();
  static auto convolution_descriptor = getConvolutionDescriptor<padding, stride, T>();
  static auto convolution_algorithm = getConvolutionAlgo(handle, input_descriptor, output_descriptor, kernel_descriptor, convolution_descriptor);
  // Workspace size can't determined statically. For now we dynamically
  // allocate workspaces.
  static const size_t workspace_bytes = [=]() -> size_t {
    size_t workspace_size = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        handle, input_descriptor, kernel_descriptor, convolution_descriptor,
        output_descriptor, convolution_algorithm, &workspace_size));
    return workspace_size;
  }();
  // std::cerr << "Workspace size: " << (workspace_bytes) << "B\n";

  T *d_workspace = gpu_malloc<T>(workspace_bytes / sizeof(T));
  const float alpha = 1.0f, beta = 0.0f;
  // auto activation = createActivationDescriptor(CUDNN_ACTIVATION_TANH, 0.0);
  // checkCUDNN(cudnnConvolutionBiasActivationForward(handle, &alpha, input_descriptor, input, kernel_descriptor, weights, convolution_descriptor, convolution_algorithm, d_workspace, workspace_bytes, &beta, output_descriptor, output, bias_descriptor, bias, activation, output_descriptor, output));
  checkCUDNN(cudnnConvolutionForward(
      handle, &alpha, input_descriptor, input, kernel_descriptor, weights,
      convolution_descriptor, convolution_algorithm, d_workspace,
      workspace_bytes, &beta, output_descriptor, output));
  const float alpha2 = 1.0f;
  checkCUDNN(cudnnAddTensor(handle, &alpha, bias_descriptor, bias, &alpha2, output_descriptor, output));
  gpu_free(d_workspace);
}

template <size_t N, size_t C, size_t H, size_t W, typename T>
void batchnorm_forward(cudnnHandle_t handle,
                       const T *src, T *avg,
                       T *variance,
                       const T *scale_shift,
                       T *dst, T *resultSaveMean, T *resultSaveInvVariance) {

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto src_desc = getTensor4dDescriptor<N, C, H, W, T>();
  auto dst_desc = getTensor4dDescriptor<N, C, H, W, T>();
  auto scale_shift_desc = getTensor4dDescriptor<1, C, 1, 1, T>();
  auto error = cudnnBatchNormalizationForwardTraining(handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, src_desc, src, dst_desc, dst, scale_shift_desc, scale_shift, scale_shift + C, 0.5, avg, variance, EPSILON, resultSaveMean, resultSaveInvVariance);
  checkCUDNN(error);
}

// convert a linear index to a row index
template <typename T>
struct sum_functor
{
  int R;
  int C;
  T *arr;

  sum_functor(int _R, int _C, T *_arr) : R(_R), C(_C), arr(_arr) {};

  __host__ __device__
  T operator()(int myC){
    T sum = 0;
      for (int i = 0; i < R; i++) sum += arr[i*C+myC];
    return sum;
    }
};


// mat should be a rows x cols matrix, vec should be a cols vector.
// Compute the sum of rows of the matrix, store it in vec
template <size_t R, size_t C, typename T>
void sum_rows(T *mat, T *_vec) {
  static_assert(C > 0, "The matrix should be a wellformed 2D matrix");
  thrust::device_ptr<T> vec{_vec};
  thrust::transform(vec, vec+C, vec, sum_functor<T>(R, C, mat));
  
}

template <size_t N, size_t IC, typename T>
void logsoftmax_forward(cudnnHandle_t handle,
                        const T *src, T *dst) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto&& src_desc = getTensor4dDescriptor<N, IC, 1, 1, T>();
  auto error = cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, src_desc, src, &beta, src_desc, dst);
  checkCUDNN(error);
}

template <size_t N, size_t IC, typename T, typename Idx>
T nll_loss(const T *src, const Idx *label) {
  auto losses = thrust::make_transform_iterator(
    thrust::counting_iterator<int>(0), 
    [src, label]__host__ __device__(int idx) -> const T& { return src[idx * IC + label[idx]]; }
    );
  auto ret = thrust::reduce(thrust::device, losses, losses+N);
  return -ret;
}

template <size_t N, size_t IC, typename T, typename Idx>
void nll_loss_backward(const T *diff_dst, const Idx *label, T *diff_src) {
  auto losses = thrust::make_transform_iterator(
    thrust::counting_iterator<int>(0), 
    [diff_src, label] __host__ __device__(int idx) -> T& { return diff_src[idx * IC + label[idx]]; }
    );
  thrust::transform(thrust::device, losses, losses+N, losses, [=]__host__ __device__(T t) { return t -(*diff_dst)/N; });
}

template <size_t N, size_t C, size_t H, size_t W, typename T>
void batchnorm_backward(cudnnHandle_t handle,
                        const T *src, T *diff_src,
                        const T *diff_dst, const T *avg,
                        const T *variance,
                        const T *gamma_beta,
                        T *diff_gamma_beta,
                        T *resultSaveMean, T *resultSaveInvVariance) {
  const float alpha = 1.0f;
  const float beta = 1.0f;
  auto src_desc = getTensor4dDescriptor<N, C, H, W, T>();
  auto scale_shift_desc = getTensor4dDescriptor<1, C, 1, 1, T>();
  auto error = cudnnBatchNormalizationBackward(handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, &alpha, &beta, src_desc, src, src_desc, diff_dst, src_desc, diff_src, scale_shift_desc, gamma_beta, diff_gamma_beta, diff_gamma_beta+C, EPSILON, resultSaveMean, resultSaveInvVariance);
  checkCUDNN(error);
}

template <size_t N, size_t C, size_t H, size_t W, size_t OC, size_t KernelSize,
          size_t padding, size_t stride, typename T>
void convolution_backward(cudnnHandle_t handle,
                          const T *diff_dst, const T *src,
                          T *diff_weights,
                          T *diff_bias) {
  const float alpha = 1.0f;
  const float beta = 1.0f;
  auto src_desc = getTensor4dDescriptor<N, C, H, W, T>();
  auto dst_desc = getTensor4dDescriptor<N, OC, H, W, T>();
  auto conv_desc = getConvolutionDescriptor<padding, stride, T>();
  auto kernel_descriptor = getFilter4dDescriptor<OC, C, KernelSize, KernelSize, T>();
  auto bias_descriptor = getTensor4dDescriptor<1, OC, 1, 1, T>();
  // cudnnConvolutionBwdDataAlgo_t algorithm{};
  // cudnnGetConvolutionBackwardDataAlgorithm(handle, kernel_descriptor, dst_desc, conv_desc, src_desc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, algorithm);

  auto error = cudnnConvolutionBackwardFilter(handle, &alpha, src_desc, src, dst_desc, diff_dst, conv_desc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, NULL, 0, &beta, kernel_descriptor, diff_weights);
  checkCUDNN(error);
  error = cudnnConvolutionBackwardBias(handle, &alpha, dst_desc, diff_dst, &beta, bias_descriptor, diff_bias);
  checkCUDNN(error);
}

template <size_t N, size_t C, size_t H, size_t W, size_t OC, size_t KernelSize,
size_t padding, size_t stride, typename T>
void convolution_backward_data(cudnnHandle_t handle, const T *diff_dst, const T *weights, const T *diff_src) {
  const float alpha = 1.0f;
  const float beta = 1.0f;
  auto kernel_desc = getFilter4dDescriptor<OC, C, KernelSize, KernelSize, T>();
  auto dst_desc = getTensor4dDescriptor<N, OC, H, W, T>();
  auto conv_desc = getConvolutionDescriptor<padding, stride, T>();
  auto src_desc = getTensor4dDescriptor<N, C, H, W, T>();
  auto error = cudnnConvolutionBackwardData(handle, &alpha, kernel_desc, weights, dst_desc, diff_dst, conv_desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, NULL, 0, &beta, src_desc, diff_src);
  checkCUDNN(error);
}

} // namespace gpu
#endif