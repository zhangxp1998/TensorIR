package scala.tensor.ir.backend

import lms.core.Backend
import lms.core.Backend.{Const, Node}

trait CPUDiffTensorCodeGen extends CPUTensorCodeGen {
  registerHeader("\"tensor_diff.h\"")
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
    case Node(s, "batchNorm-backprop", List(Const(dims: Seq[Int]), src, diff_dst, avg, variance, diff_src, gamma_beta, diff_gama_beta), _) =>
      val Seq(n, c, h, w) = dims
      emit(s"batchnorm_backward<$n, $c, $h, $w>(eng, stream, ")
      shallow(src)
      emit(", ")
      shallow(diff_src)
      emit(", ")
      shallow(diff_dst)
      emit(", ")
      shallow(avg)
      emit(", ")
      shallow(variance)
      emit(", ")
      shallow(gamma_beta)
      emit(", ")
      shallow(diff_gama_beta)
      emit(")")
    case Node(s, "conv2d-backprop", List(Const(Seq(n, c, h, w)), Const(Seq(oc, kh, padding, stride)), diff_dst, src, diff_weight, diff_bias), _)=>
      emit(s"convolution_backward<$n, $c, $h, $w, $oc, $kh, $padding, $stride>(eng, stream, ")
      shallow(diff_dst)
      emit(", ")
      shallow(src)
      emit(", ")
      shallow(diff_weight)
      emit(", ")
      shallow(diff_bias)
      emit(")")
    case _ => super.shallow(n)
  }
  def emitStubComment(op: String): Unit = emit(s"/*Stub for ${op}*/")
}