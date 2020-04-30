package tensor.ir.backend

import lms.core.{Backend, Graph}
import lms.core.Backend.{Block, Const, Node}
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
    case _ => super.shallow(node)
  }
  override val transformFuncName = "gpu::transform"
  override val fillFuncName = "gpu::fill"
  override val sgemmFuncName = "gpu::sgemm"
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