package tensor.ir

import lms.core.Backend
import lms.core.stub.Adapter
import lms.macros.SourceContext
import tensor.ir.backend.GPUTensorDriverC

trait GPUTensorOps extends CPUTensorOps {
  def createMemDesc(dims: Seq[Int]): Rep[MemDesc] = {
    Wrap[MemDesc](Adapter.g.reflect("mem-desc", Backend.Const(dims)))
  }
  object GPUTensor {
    def apply[A: Manifest: Ordering](xs: Seq[Int], allocType: AllocationType)(implicit pos: SourceContext): GPUTensor[A] = {
      new GPUTensor[A](xs, allocType)
    }
  }
  class GPUTensor[A: Manifest : Ordering](override val dims: Seq[Int], override val allocType: AllocationType) extends
    Tensor[A](dims, allocType) {
    override lazy val memDesc: Rep[MemDesc] = createMemDesc(dims)
  }
}

object GPUTensorOps {
  def main(args: Array[String]): Unit = {
    val dslDriver = new GPUTensorDriverC[String,Unit] {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        val x = GPUTensor[Float](Seq(10), AllocationType.Data)
        x.fill(10.0f)
//        x(Seq(0)) = 2.0f
        println(x(0))
      }
    }
    dslDriver.eval("0")
  }
}