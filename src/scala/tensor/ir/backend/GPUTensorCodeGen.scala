package tensor.ir.backend

import lms.core.{Backend, Graph}
import lms.core.Backend.{Block, Const, Def, Node}
import lms.core.stub.{DslDriverC, DslGenC}
import lms.core.utils.time
import tensor.ir.{CPUTensorOps, GPUTensorOps, RandomOpsCodegen}

import scala.tensor.ir.backend.CPUDiffTensorCodeGen

trait GPUTensorCodeGen extends CPUDiffTensorCodeGen {

  // GPU memory planning requires us to be aware of 2 heaps
  // TODO write a multi-heap memory planner
  override def memoryPlanning(g: Graph): Graph = {
    g
  }

  override def registerRuntimeLibHeaders(): Unit = {
    registerHeader("\"gpu_tensor.h\"")
  }

  registerTopLevelFunction("cleanup") {
    emitln("void cleanup() {}")
  }
  override def emitEngine(): Unit = {
  }
  override def emitStream(): Unit = {

  }
  override def shallow(node: Node): Unit = node match {
    case Node(_, "tensor-new", Const(manifest: Manifest[_])::Backend.Const(dims: Seq[Int])::Const(_)::Nil, _) =>
      emit(s"gpu::gpu_malloc<${remap(manifest)}>(${dims.product})")
    case Node(s, "tensor-apply", List(_, tensor, Const(idx: Seq[Int]), Const(dims: Seq[Int])), _) =>
      val sizes = dims.scanRight(1)(_ * _).tail
      emit(s"gpu::read_gpu_mem(")
      shallow(tensor)
      emit(s", ${idx.zip(sizes).map{case (a, b) => a*b}.sum})")
    case Node(s, "tensor-update", List(_, tensor, Const(idx: Seq[Int]), newVal, Const(dims: Seq[Int])), _) =>
      val sizes = dims.scanRight(1)(_ * _).tail
      emit(s"gpu::write_gpu_mem(")
      shallow(tensor)
      emit(s", ${idx.zip(sizes).map{case (a, b) => a*b}.sum}, ")
      shallow(newVal)
      emit(")")
    case Node(s, "tensor-transform-range", List(Const(mA: Manifest[_]), data, block: Block, begin, end), _) =>
      assert(block.in.length == 1)
      emit("gpu::transform(")
      emitBeginEnd(data, begin, end)
      emit(", ")
      shallow(data)
      emit("+")
      shallow(begin)
      emit(s", [=] __device__ __host__ (${remap(mA)} ")
      shallow(block.in.head)
      emit(")")
      quoteBlockPReturn(traverse(block))
      emit(")")
    case Node(_, "tensor-copy", List(Const(mA: Manifest[_]), tensor, Const(dims: Seq[Int]), Const(allocType)), _) =>
      val totalSize = dims.product
      emit(s"gpu::memdup<${remap(mA)}>(")
      shallow(tensor)
      emit(", ")
      emit(totalSize.toString)
      emit(")")
    case Node(s, "tensor-convolution2d", List(mA, input, output, kernels, bias, Const(Seq(n, c, h, w)), Const(Seq(oc, kh, padding, stride))), _) =>
      emit(s"gpu::conv2d_forward<$n, $c, $h, $w, $oc, $kh, $padding, $stride>(gpu::cudnnHandle, ")
      shallowParams(input, output, kernels, bias)
      emit(")")
    case Node(s, "batchnorm-forward", List(Const(mA: Manifest[_]), Const(dims: Seq[Int]), Const(epsilon: Float), src, avg, variance, gamma_beta, dst), _) =>
      // TODO support custom epsilon
      val Seq(n, c, h, w) = dims
      emit(s"gpu::batchnorm_forward<$n, $c, $h, $w, ${remap(mA)}>(gpu::cudnnHandle, ")
      shallowParams(src, avg, variance, gamma_beta, dst)
      emit(", NULL, NULL)")
    case Node(s, "mem-desc", List(Const(mA: Manifest[_]), data, Const(dims: Seq[Int])), _) =>
      assert(dims.length <= 4, "Tensors with >4 dimensions are not supported")
      // Padd dims with 1s if length less than 4
      shallow(data)
//      val padded = (dims ++ Seq(1, 1, 1, 1)).take(4)
//      emit(s"gpu::createTensor4dDescriptor<${remap(mA)}>(${padded.mkString(",")})")
    case _ => super.shallow(node)
  }
  override def remap(mA: Manifest[_]) = mA.toString match {
    case s if s.contains("MemDesc") =>
      assert(mA.typeArguments.length == 1, s"Expect MemDesc to have exactly 1 type parameter ${mA.typeArguments}")
      remap(mA.typeArguments.head) + " *"
    case _ => super.remap(mA)
  }

  override val forwardFuncNames = Map(
    "matrix-multiply" -> "gpu::sgemm",
    "tensor-fill" -> "gpu::fill",
    "tensor-binary-transform-range" -> "gpu::transform",
  )

  override def getPrimitiveOpLambda(op: String, mA: Manifest[_]): String = op match {
    case "+" => s"thrust::plus<${remap(mA)}>()"
    case "-" => s"thrust::minus<${remap(mA)}>()"
    case "*" => s"thrust::multiplies<${remap(mA)}>()"
    case "/" => s"thrust::divides<${remap(mA)}>()"
    case "%" => s"thrust::modulus<${remap(mA)}>()"
    case "==" => s"thrust::equal_to<${remap(mA)}>()"
    case "!=" => s"thrust::not_equal_to<${remap(mA)}>()"
  }
}

abstract class GPUTensorDriverC[A: Manifest, B: Manifest] extends DslDriverC[A, B] with GPUTensorOps { q =>
  override val codegen = new GPUTensorCodeGen {
    override val IR: q.type = q
  }
  override val outputSrcPath = "gen/snippet.cu"
  override val outputBinPath = "gen/build/snippet_gpu"
}