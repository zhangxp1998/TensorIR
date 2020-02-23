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
  test("dropout") {
    val length = 400
    val p = 0.5f
    val dslDriver: TensorDriverC[String, Unit] = new TensorDriverC[String,Unit] {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor[Float](Seq(length))
        x.mapInplaceWithFlatIdx(idx => idx)
        val output = x.dropout(p)
        for (i <- 0 until length: Rep[Range]) {
          println(output.unsafe_apply(i))
        }
      }
    }

    val res = dslDriver.eval("0")
    var sum = 0
    res.map(_.toFloat).zipWithIndex foreach { case (value, idx) =>
      assert(value == 0 || value == idx/p)
      if (value == 0) sum += 1
    }
    // With N=400, this will success 99.7% of time
    assert(Math.abs(sum.toFloat/length - p)/p <= 0.075f)
  }
}
