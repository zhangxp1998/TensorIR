package tensor.ir.backend

import lms.core.Backend
import lms.core.Backend.{Const, Node}
import lms.core.stub.{DslDriverC, DslExp}
import tensor.ir.{GPUTensorDiff, GPUTensorOps}

import scala.tensor.ir.backend.CPUDiffTensorCodeGen

trait GPUDiffTensorCodeGen extends GPUTensorCodeGen {
  override val backpropFuncNames = Map(
    "matmul-backprop" -> "gpu::matmul_backprop"
  )

  override def shallow(node: Backend.Node): Unit = node match {
    case Node(s, "batchNorm-backprop", List(Const(mA: Manifest[_]), Const(dims: Seq[Int]), src, diff_dst, avg, variance, diff_src, gamma_beta, diff_gama_beta), _) =>
      val Seq(n, c, h, w) = dims
      emit(s"gpu::batchnorm_backward<$n, $c, $h, $w, ${remap(mA)}>(gpu::cudnnHandle, ")
      shallowParams(src, diff_src, diff_dst, avg, variance, gamma_beta, diff_gama_beta)
      emit(", NULL, NULL)")
    case Node(s, "conv2d-backprop", List(Const(mA: Manifest[_]), Const(Seq(n, c, h, w)), Const(Seq(oc, kh, padding, stride)), diff_dst, src, diff_weight, diff_bias), _) =>
      emit(s"gpu::convolution_backward<$n, $c, $h, $w, $oc, $kh, $padding, $stride, ${remap(mA)}>(gpu::cudnnHandle, ")
      shallowParams(diff_dst, src, diff_weight, diff_bias)
      emit(")")
    case Node(s, "conv2d-data-backprop", List(Const(mA: Manifest[_]), Const(Seq(n, c, h, w)), Const(Seq(oc, kh, padding, stride)), diff_dst, weight, diff_src), _) =>
      emit(s"gpu::convolution_backward_data<$n, $c, $h, $w, $oc, $kh, $padding, $stride, ${remap(mA)}>(gpu::cudnnHandle, ")
      shallowParams(diff_dst, weight, diff_src)
      emit(")")
    case _ => super.shallow(node)
  }
}

abstract class GPUTensorDiffDriverC[A: Manifest, B: Manifest] extends DslDriverC[A, B] with GPUTensorDiff { q =>
  override val codegen = new GPUDiffTensorCodeGen() {
    override val IR: q.type = q
  }
  override val outputSrcPath = "gen/snippet.cu"
  override val outputBinPath = "gen/build/snippet_gpu"
}