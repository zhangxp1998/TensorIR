package tensor.ir

import org.scalatest.FunSuite
import lms.macros.SourceContext

class TensorDifferentiationTest extends FunSuite {
  test("add") {
    val dslDriver = new TensorDriverC[String,Unit] with TensorDifferentiation {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        val length = 20
        val x = Tensor[Float](Seq(length))
        x.mapInplaceWithFlatIdx((_, idx) => idx)
        val gradient = TensorR.grad(a => a*a*a)(x)
        for (i <- 0 until length : Rep[Range]) {
          println(gradient.data(i))
        }
      }
    }

    val res = dslDriver.eval("5")
    assert(res.map(_.toDouble).zipWithIndex.forall{case (g, idx) => g == idx*idx*3})
  }
}
