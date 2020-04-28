#include "gpu_tensor.h"
cublasHandle_t gpu::cublasHandle = gpu::createCublasHandle();


cublasHandle_t gpu::createCublasHandle() {
    cublasHandle_t handle;
    auto error = cublasCreate(&handle);
    assert(CUBLAS_STATUS_SUCCESS == error);
    return handle;
}

void gpu::sgemm(const char transA, const char transB, const float *a, const float *b,
           float *c, const size_t M, const size_t K, const size_t N,
           const float& alpha, const float& beta) {

  int64_t lda = tolower(transA) == 'n' ? K : M;
  int64_t ldb = tolower(transB) == 'n' ? N : K;
  int64_t ldc = N;

  auto&& opA = tolower(transA) == 'n' ? CUBLAS_OP_N : CUBLAS_OP_T;
  auto&& opB = tolower(transB) == 'n' ? CUBLAS_OP_N : CUBLAS_OP_T;
  // CUBLAS expects data in column major format
  cublasSgemm(cublasHandle, opA, opB, M, N, K, &alpha, a, lda, b, ldb, &beta, c, ldc);
}
