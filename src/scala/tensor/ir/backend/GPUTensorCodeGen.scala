package tensor.ir.backend

import lms.core.{Backend, Graph}
import lms.core.Backend.{Const, Node}
import lms.core.stub.{DslDriverC, DslGenC}
import lms.core.utils.time
import tensor.ir.{CPUTensorOps, GPUTensorOps, RandomOpsCodegen}

import scala.tensor.ir.backend.CPUTensorCodeGen

trait GPUTensorCodeGen extends PrintfCodeGen {

  // GPU memory planning requires us to be aware of 2 heaps
  // TODO write a multi-heap memory planner
  def memoryPlanning(g: Graph): Graph = {
    g
  }

  registerHeader("\"gpu_tensor.h\"")
  registerTopLevelFunction("cleanup") {
    emitln("void cleanup() {}")
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
    case Node(_, "tensor-fill", List(mA, tensor, fillVal, Const(dims: Seq[Int])), _) =>
      val totalSize = dims.product
      emit("gpu::fill(")
      shallow(tensor)
      emit(", ")
      shallow(tensor)
      emit(s" + $totalSize, ${quote(fillVal)})")
    case _ => super.shallow(node)
  }
}

abstract class GPUTensorDriverC[A: Manifest, B: Manifest] extends DslDriverC[A, B] with GPUTensorOps { q =>
  override val codegen = new GPUTensorCodeGen {
    override val IR: q.type = q
  }
  override lazy val f: A => Stream[String] = {
    // TBD: should read result of type B?
    val outputBinPath = "gen/build/snippet_gpu"
    val out = new java.io.PrintStream("gen/snippet.cu")
    out.println(code)
    out.close()
    (new java.io.File(outputBinPath)).delete
    import scala.sys.process._
    time("cmake") { (s"cmake -Bgen/build -Sgen -DCMAKE_BUILD_TYPE=Debug": ProcessBuilder).lineStream.foreach(Console.println) }
    time("clang++") { (s"cmake --build gen/build": ProcessBuilder).lineStream.foreach(Console.println) }
    (a: A) => (s"$outputBinPath $a": ProcessBuilder).lineStream
  }
}