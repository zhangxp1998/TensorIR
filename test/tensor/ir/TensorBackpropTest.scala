package tensor.ir

import org.scalatest.FunSuite
import lms.macros.SourceContext

class TensorBackpropTest extends FunSuite {
  test("softmax-backprop") {
    val rows = 10
    val dslDriver = new CPUTensorDiffDriverC[String, Unit] {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor[Float](Seq(rows, rows), AllocationType.Data)
        x.mapInplaceWithFlatIdx(idx => idx % rows + 1)
        val labels = Tensor[Int](Seq(rows), AllocationType.Data)
        labels.mapInplaceWithFlatIdx(idx => idx % rows)

        def backward(x: Tensor[Float]): Tensor[Float] = TensorR.grad(a => a.softmaxLoss(labels))(x)

        val gradients = backward(x)
        for (i <- 0 until gradients.dims.product: Rep[Range]) {
          println(gradients.unsafe_apply(i))
        }
        ()
      }
    }
    val res = dslDriver.eval("0")

    val source = scala.io.Source.fromFile("data/softmax_backprop_expected.txt")
    val ans = try source.getLines().map(_.toDouble).toList finally source.close()

    val actual = res.map(_.toDouble).toList
    actual.zip(ans).foreach{ case (act, expected) =>
      assert(Math.abs(act-expected) <= 1e-6)
    }
  }
}
