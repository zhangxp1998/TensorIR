#include "tensor.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include "mpi_helper.h"

void matmul_backprop(const float *m1, const float *m2, const float *y,
                     float *d1, float *d2, const size_t M, const size_t K,
                     const size_t N) {
  // m1: M*K, m2: K*N, y: M*N
  // d1 += y * m2.T => M*N x N*K = M*K
  // d2 += m1.T * y => K*M x M*N = K*N
  sgemm('N', 'T', y, m2, d1, M, N, K, 1.0f, 1.0f);
  sgemm('T', 'N', m1, y, d2, K, M, N, 1.0f, 1.0f);
}

void sgemm(const char transA, const char transB, const float *a, const float *b,
           float *c, const size_t M, const size_t K, const size_t N,
           float alpha, float beta) {
  // Prepare leading dimensions
  int64_t lda = tolower(transA) == 'n' ? K : M;
  int64_t ldb = tolower(transB) == 'n' ? N : K;
  int64_t ldc = N;
  auto error = dnnl_sgemm(transA, transB, M, N, K, alpha, a, lda, b, ldb, beta, c, ldc);
  assert(error == dnnl_success);
}


void cleanup() noexcept {
  MPI_Finalize();
  std::cerr << "Time spent in MPI calls: " << mpi_duration.count() << "ms\n";
}
