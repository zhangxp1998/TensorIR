package tensor.ir

import org.scalatest.FunSuite
import lms.macros.SourceContext

class TensorOpsTest extends FunSuite {
  test("relu") {
    val length = 20
    val dslDriver = new TensorDriverC[String,Unit] {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor[Float](Seq(length))
        x.mapInplaceWithFlatIdx(idx => idx - length/2)
        val output = x.relu()
        for (i <- 0 until length: Rep[Range]) {
          println(output.unsafe_apply(i))
        }
      }
    }

    val res = dslDriver.eval("0")
    res.map(_.toFloat).zipWithIndex.map{case (value, idx) =>
      assert(value == Math.max(0, idx-length/2))
    }
  }
}
