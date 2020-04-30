package tensor.ir.backend

import lms.core.Backend
import lms.core.stub.{DslDriverC, DslExp}
import tensor.ir.{GPUTensorDiff, GPUTensorOps}

import scala.tensor.ir.backend.CPUDiffTensorCodeGen

trait GPUDiffTensorCodeGen extends GPUTensorCodeGen {
  override val backpropFuncNames = Map(
    "matmul-backprop" -> "gpu::matmul_backprop"
  )
}

abstract class GPUTensorDiffDriverC[A: Manifest, B: Manifest] extends DslDriverC[A, B] with GPUTensorDiff { q =>
  override val codegen = new GPUDiffTensorCodeGen() {
    override val IR: q.type = q
  }
  override val outputSrcPath = "gen/snippet.cu"
  override val outputBinPath = "gen/build/snippet_gpu"
}