package tensor.ir

import lms.core.Backend
import lms.core.stub.Adapter
import lms.macros.SourceContext
import tensor.ir.backend.GPUTensorDriverC

trait GPUTensorOps extends CPUTensorDiff {
  def createMemDesc(dims: Seq[Int]): Rep[MemDesc] = {
    Wrap[MemDesc](Adapter.g.reflect("mem-desc", Backend.Const(dims)))
  }
  object GPUTensor {
    def apply[A: Manifest: Ordering](xs: Seq[Int], allocType: AllocationType)(implicit pos: SourceContext): GPUTensor[A] = {
      new GPUTensor[A](xs, allocType)
    }
    def fill[A: Manifest: Ordering](dims: Seq[Int], fillVal: A, allocType: AllocationType): GPUTensor[A] = {
      val tensor = new GPUTensor[A](dims, allocType)
      tensor.fill(fillVal)
      tensor
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
        x.transform(x => x/2.0f)
        val y = GPUTensor[Float](Seq(10), AllocationType.Data)
        y.fill(5.0f)
        println(x(0))
        val z = x.add(y)
        println(z(0))
        val a = GPUTensor.fill[Float](Seq(1, 10), 2, AllocationType.Data)
        val b = GPUTensor.fill[Float](Seq(10, 1), 3, AllocationType.Data)
        val c = a.matmul(b, None, AllocationType.Data)
        println(c(0, 0))
        b(Seq(2, 0)) = 20.0f
        b(Seq(3, 0)) = 30.0f
//        a(Seq(0, 2)) = 4.0f
//        a(Seq(0, 3)) = 5.0f

        val d = b.matmul(a, None, AllocationType.Data)
        println(b(2, 0)*a(0, 3), d(2, 3))
        println(b(3, 0)*a(0, 2), d(3, 2))
      }
    }
    dslDriver.eval("0")
  }
}