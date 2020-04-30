package tensor.ir

import tensor.ir.backend.GPUTensorDiffDriverC
import lms.macros.SourceContext

trait GPUTensorDiff extends GPUTensorOps {

}

object GPUTensorDiff {
  def main(args: Array[String]): Unit = {
    val dslDriver = new GPUTensorDiffDriverC[String,Unit] {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        val x = GPUTensor[Float](Seq(1, 10), AllocationType.Data)
        val y = GPUTensor[Float](Seq(10, 1), AllocationType.Data)
        x.fill(2.0f)
        y.fill(3.0f)
        val (grad_x, grad_y) = TensorR.grad((a, b) => a.matmul(b))(x, y)
        println(grad_x(0, 0), grad_y(0, 0))
      }
    }
    dslDriver.eval("0")
  }
}