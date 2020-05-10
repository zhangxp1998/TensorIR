package scala.tensor.ir.backend

import lms.core.Backend
import lms.core.Backend.{Const, Node}

trait CPUDiffTensorCodeGen extends CPUTensorCodeGen {
  override def registerRuntimeLibHeaders(): Unit = {
    super.registerRuntimeLibHeaders()
    registerHeader("\"tensor_diff.h\"")
  }
  val backpropFuncNames = Map(
    "matmul-backprop" -> "matmul_backprop"
  )
  override def shallow(node: Node): Unit = node match {
    case Node(s, "matmul-backprop", List(m1, m2, diff_dst, d1, d2, Const(Seq(m: Int, k: Int, n: Int))), _) =>
      emit(s"${backpropFuncNames(node.op)}(")
      shallowParams(m1, m2, diff_dst, d1, d2)
      emit(s", $m, $k, $n)")
    case Node(s, "conv-backprop", List(x, kernel, output, d, kernelD, outputD, Const(Seq(padding: Int, stride: Int))), _) =>
      // TODO implement convolution backprop
      emitStubComment(node.op)
    case Node(s, "batchNorm-backprop", List(Const(mA: Manifest[_]), Const(dims: Seq[Int]), src, diff_dst, avg, variance, diff_src, gamma_beta, diff_gama_beta), _) =>
      val Seq(n, c, h, w) = dims
      emit(s"batchnorm_backward<$n, $c, $h, $w>(eng, stream, ")
      shallowParams(src, diff_src, diff_dst, avg, variance, gamma_beta, diff_gama_beta)
      emit(")")
    case Node(s, "conv2d-backprop", List(Const(mA: Manifest[_]), Const(Seq(n, c, h, w)), Const(Seq(oc, kh, padding, stride)), diff_dst, src, diff_weight, diff_bias), _) =>
      emit(s"convolution_backward<$n, $c, $h, $w, $oc, $kh, $padding, $stride>(eng, stream, ")
      shallowParams(diff_dst, src, diff_weight, diff_bias)
      emit(")")
    case Node(s, "conv2d-data-backprop", List(Const(mA: Manifest[_]), Const(Seq(n, c, h, w)), Const(Seq(oc, kh, padding, stride)), diff_dst, weight, diff_src), _) =>
      emit(s"convolution_backward_data<$n, $c, $h, $w, $oc, $kh, $padding, $stride>(eng, stream, ")
      shallowParams(diff_dst, weight, diff_src)
      emit(")")
    case Node(s, "logsoftmax-backward", List(mA, diff_dst, dst, diff_src, Const((rows: Int, rowSize: Int))), _) =>
      emit(s"logsoftmax_backward<$rows, $rowSize>(eng, stream, ")
      shallowParams(diff_dst, dst, diff_src)
      emit(")")
    case _ => super.shallow(node)
  }
  def emitStubComment(op: String): Unit = emit(s"/*Stub for ${op}*/")
}