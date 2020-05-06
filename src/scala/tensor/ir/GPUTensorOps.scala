package tensor.ir

import lms.core.Backend
import lms.core.stub.Adapter
import lms.macros.SourceContext
import tensor.ir.backend.GPUTensorDriverC

trait GPUTensorOps extends CPUTensorDiff {
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
    assert(dims.length <= 4, "Tensors with >4 dimensions are not supported")
//    override lazy val memDesc: Rep[MemDesc] = createMemDesc[A](dims)
  }
}

object GPUTensorOps {
  def main(args: Array[String]): Unit = {
    val dslDriver = new GPUTensorDriverC[String,Unit] {
      def convTest(): Unit = {
        val input = GPUTensor[Float](Seq(32, 3, 28, 28), AllocationType.Data)
        val kernel = GPUTensor[Float](Seq(3, 3, 3, 3), AllocationType.Parameter)
        val bias = GPUTensor[Float](Seq(3), AllocationType.Parameter)
        input.fill(1.0f)
        kernel.fill(0.1f)
        bias.fill(2.0f)
        val output = input.conv2d(kernel, bias, 1, 1)
        println(output(0, 0, 0, 0))
      }
      def matmulTest(): Unit = {
        val a = GPUTensor.fill[Float](Seq(1, 10), 2, AllocationType.Data)
        val b = GPUTensor.fill[Float](Seq(10, 1), 3, AllocationType.Data)
        val c = a.matmul(b, None, AllocationType.Data)
        println(c(0, 0))
        b(Seq(2, 0)) = 20.0f
        b(Seq(3, 0)) = 30.0f

        val d = b.matmul(a, None, AllocationType.Data)
        println(b(2, 0)*a(0, 3), d(2, 3))
        println(b(3, 0)*a(0, 2), d(3, 2))
      }
      def basicOpsTest(): Unit = {
        val x = GPUTensor[Float](Seq(10), AllocationType.Data)
        x.fill(10.0f)
        x.transform(x => x/2.0f)
        val y = GPUTensor[Float](Seq(10), AllocationType.Data)
        y.fill(5.0f)
        println(x(0))
        val z = x.add(y)
        println(z(0))
      }
      def batchNormTest(): Unit = {
        val x = Tensor[Float](Seq(32, 3, 28, 28), AllocationType.Data)
        x.fill(0.0f)
        val gamma_beta = Tensor.fill[Float](Seq(2, 3), 1.0f, AllocationType.Data)
        val (dst, avg, variance) = x.batchNorm(gamma_beta)
        println(dst(0, 0, 0, 0), avg(0), variance(0))
      }
      def sumRowsTest(): Unit = {
        val a = Tensor.fill[Float](Seq(6, 7), 1.0f, AllocationType.Data)
        val b = Tensor[Float](Seq(7), AllocationType.Data)
        a.sumRows(b)
        println(b(0), b(1), b(2))
      }
      override def snippet(x: Rep[String]): Rep[Unit] = {
        val rows = 10
        val x = Tensor[Float](Seq(rows, rows), AllocationType.Data)
        x.mapInplaceWithFlatIdx(idx => idx % rows + 1)
        val labels = Tensor[Int](Seq(rows), AllocationType.Data)
        labels.mapInplaceWithFlatIdx(idx => idx % rows)
        println(x(0, 0), x(0, 1), x(0, 2))
        val loss = x.softmaxLoss(labels)
        println(loss)
      }
    }
    dslDriver.eval("0")
  }
}