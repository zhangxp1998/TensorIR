package scala.tensor.ir.backend

import lms.core.Backend
import lms.core.Backend.{Const, Node}

trait CPUDiffTensorCodeGen extends CPUTensorCodeGen {
  registerTopLevelFunction("matmul_backprop") {
    emit(
      """
        |void matmul_backprop(const float *m1, const float *m2, const float *y,
        |     float *d1, float *d2, const size_t M, const size_t K, const size_t N) {
        |  // m1: M*K, m2: K*N, y: M*N
        |  // d1 += y * m2.T => M*N x N*K = M*K
        |  // d2 += m1.T * y => K*M x M*N = K*N
        |  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, y, M, m2, K, 1.0f, d1, M);
        |  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, M, N, 1.0f, m1, M, y, M, 1.0f, d2, M);
        |}
        |""".stripMargin)
  }
  override def shallow(n: Node): Unit = n match {
    case Node(s, "matmul-backprop", List(m1, m2, y, d1, d2, Backend.Const(Seq(m: Int, k: Int, n: Int))), _) =>
      emit("matmul_backprop(")
      shallow(m1)
      emit(", ")
      shallow(m2)
      emit(", ")
      shallow(y)
      emit(", ")
      shallow(d1)
      emit(", ")
      shallow(d2)
      emit(s", $m, $k, $n)")
    case Node(s, "conv-backprop", List(x, kernel, output, d, kernelD, outputD, Backend.Const(Seq(padding: Int, stride: Int))), _) =>
      // TODO implement convolution backprop
      emitStubComment(n.op)
    case Node(s, "batchNorm-backprop", List(Const(dims: Seq[Int]), src, diff_dst, avg, variance, diff_src), _) =>
      // TODO implement batchnorm backprop
      emitStubComment(n.op)
    case Node(s, "conv2d-backprop", x::y_x::d::y_d::Backend.Const(Seq(padding: Int, stride: Int)):: gradients, _)=>
      // TODO implement conv2d backprop
      emitStubComment(n.op)
    case _ => super.shallow(n)
  }
  def emitStubComment(op: String): Unit = emit(s"/*Stub for ${op}*/")
}