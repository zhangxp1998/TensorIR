#include "gpu_tensor.h"
cublasHandle_t gpu::cublasHandle = gpu::createCublasHandle();
cudnnHandle_t gpu::cudnnHandle = gpu::createCudnnHandle();
curandGenerator_t gpu::gen = gpu::createCudaRandGenerator();

constexpr auto CUDA_DEVICE = 0;
curandGenerator_t gpu::createCudaRandGenerator() {
  curandGenerator_t gen;
  /* Create pseudo-random number generator */
  CURAND_CALL(curandCreateGenerator(&gen, 
              CURAND_RNG_PSEUDO_DEFAULT));
  
  /* Set seed */
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 
              1234ULL));
  return gen;
}

cublasHandle_t gpu::createCublasHandle()
{
  cudaSetDevice(CUDA_DEVICE);
  cublasHandle_t handle{};
  auto error = cublasCreate(&handle);
  assert(CUBLAS_STATUS_SUCCESS == error);
  return handle;
}

cudnnHandle_t gpu::createCudnnHandle()
{
  cudaSetDevice(CUDA_DEVICE);
  cudnnHandle_t cudnn;
  checkCUDNN(cudnnCreate(&cudnn));
  return cudnn;
}

cudnnActivationDescriptor_t gpu::createActivationDescriptor(cudnnActivationMode_t activationMode, double coef)
{
  cudnnActivationDescriptor_t activation{};
  checkCUDNN(cudnnCreateActivationDescriptor(&activation));
  cudnnSetActivationDescriptor(activation, activationMode, CUDNN_PROPAGATE_NAN, coef);
  return activation;
}

cudnnConvolutionFwdAlgo_t gpu::getConvolutionAlgo(cudnnHandle_t handle, cudnnTensorDescriptor_t input_descriptor, cudnnTensorDescriptor_t output_descriptor, cudnnFilterDescriptor_t kernel_descriptor, cudnnConvolutionDescriptor_t convolution_descriptor)
{
  cudnnConvolutionFwdAlgo_t convolution_algorithm{};
  checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
      handle, input_descriptor, kernel_descriptor, convolution_descriptor,
      output_descriptor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
      /*memoryLimitInBytes=*/0, &convolution_algorithm));
  return convolution_algorithm;
}

cudnnConvolutionBwdFilterAlgo_t gpu::getConvolutionBwdFilterAlgo(cudnnHandle_t handle, cudnnTensorDescriptor_t src_desc, cudnnTensorDescriptor_t dst_desc, cudnnConvolutionDescriptor_t conv_desc, cudnnFilterDescriptor_t weight_desc) {
  cudnnConvolutionBwdFilterAlgo_t algo{};
  checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(handle, src_desc, dst_desc, conv_desc, weight_desc, CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, 0, &algo));
  return algo;
}

cudnnConvolutionBwdDataAlgo_t gpu::getConvolutionBwdDataAlgo(cudnnHandle_t handle, cudnnTensorDescriptor_t src_desc, cudnnTensorDescriptor_t dst_desc, cudnnConvolutionDescriptor_t conv_desc, cudnnFilterDescriptor_t weight_desc) {
  cudnnConvolutionBwdDataAlgo_t algo{};
  cudnnGetConvolutionBackwardDataAlgorithm(handle, weight_desc, dst_desc, conv_desc, src_desc, CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE, 0, &algo);
  return algo;
}

void gpu::gpu_free(void *p)
{
  auto error = cudaFree(p);
  assert(error == cudaSuccess);
}

static void sgemm_column_major(const char transA, const char transB, const float *a, const float *b,
                               float *c, const size_t M, const size_t K, const size_t N,
                               const float &alpha, const float &beta)
{

  int lda = tolower(transA) == 'n' ? M : K;
  int ldb = tolower(transB) == 'n' ? K : N;
  int ldc = M;
  auto &&opA = tolower(transA) == 'n' ? CUBLAS_OP_N : CUBLAS_OP_T;
  auto &&opB = tolower(transB) == 'n' ? CUBLAS_OP_N : CUBLAS_OP_T;
  auto error = cublasSgemm(gpu::cublasHandle, opA, opB, M, N, K, &alpha, a, lda, b, ldb, &beta, c, ldc);
  assert(error == CUBLAS_STATUS_SUCCESS);
}

static char invertTrans(const char transA)
{
  return tolower(transA) == 'n' ? 'T' : 'N';
}

// Expects data in row major format
void gpu::sgemm(const char transA, const char transB, const float *a, const float *b,
                float *c, const size_t M, const size_t K, const size_t N,
                const float &alpha, const float &beta)
{

  sgemm_column_major(invertTrans(transA), invertTrans(transB), b, a, c, N, K, M, alpha, beta);
}

void gpu::matmul_backprop(const float *m1, const float *m2, const float *y,
                          float *d1, float *d2, const size_t M, const size_t K,
                          const size_t N)
{
  // m1: M*K, m2: K*N, y: M*N
  // d1 += y * m2.T => M*N x N*K = M*K
  // d2 += m1.T * y => K*M x M*N = K*N
  gpu::sgemm('N', 'T', y, m2, d1, M, N, K, 1.0f, 1.0f);
  gpu::sgemm('T', 'N', m1, y, d2, K, M, N, 1.0f, 1.0f);
}

void *gpu::memcpy(void *dest, const void *src, size_t n) {
  auto error = cudaMemcpy(dest, src, n, cudaMemcpyDefault);
  assert(error == cudaSuccess);
  return dest;
}
