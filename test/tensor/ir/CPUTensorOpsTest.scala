package tensor.ir

import org.scalatest.FunSuite
import lms.macros.SourceContext

class CPUTensorOpsTest extends FunSuite {
  test("relu") {
    val length = 20
    val dslDriver = new TensorDriverC[String,Unit] {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor[Float](Seq(length), AllocationType.Data)
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
        val x = Tensor[Float](Seq(length), AllocationType.Data)
        x.mapInplaceWithFlatIdx(idx => idx+1)
        val output = x.dropout(p)
        for (i <- 0 until length: Rep[Range]) {
          println(output.unsafe_apply(i))
        }
      }
    }

    val res = dslDriver.eval("0")
    var sum = 0
    res.map(_.toFloat).zipWithIndex foreach { case (value, idx) =>
      assert(value == 0 || value == (idx+1)/p)
      if (value == 0) sum += 1
    }
    // With N=400, this will success 99.7% of time
    assert(Math.abs(sum.toFloat/length - p)/p <= 0.075f)
  }
  test("sum") {
    val length = 20
    val dslDriver = new TensorDriverC[String,Unit] {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor[Float](Seq(length), AllocationType.Data)
        x.mapInplaceWithFlatIdx(idx => idx + 1)
        println(x.sum())
      }
    }

    val res = dslDriver.eval("0")
    val nums = res.map(_.toDouble)
    assert(nums.length == 1)
    assert(nums.head == (1+length)*length/2)
  }
  test("softmaxLoss") {
    val rows = 10
    val dslDriver = new TensorDriverC[String,Unit] {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor[Float](Seq(rows, rows), AllocationType.Data)
        x.mapInplaceWithFlatIdx(idx => idx % rows + 1)
        val labels = Tensor[Int](Seq(rows), AllocationType.Data)
        labels.mapInplaceWithFlatIdx(idx => idx % rows)
        val loss = x.softmaxLoss(labels)
        println(loss)
      }
    }
    val res = dslDriver.eval("0")
    val nums = res.map(_.toDouble)
    assert(nums.length == 1)
    assert(math.abs(nums.head - 4.9586297) <= 1e-5)
  }
  test("sumRows") {
    val rows = 10
    val dslDriver = new TensorDriverC[String,Unit] {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor[Float](Seq(rows, rows), AllocationType.Data)
        x.mapInplaceWithFlatIdx(idx => idx % rows + 1)
        val dst = Tensor[Float](Seq(rows), AllocationType.Data)
        x.sumRows(dst)
        for (i <- 0 until rows: Rep[Range]) {
          println(dst.unsafe_apply(i))
        }
      }
    }
    val res = dslDriver.eval("0")
    val nums = res.map(_.toDouble)
    assert(nums.length == 10)
    nums.zipWithIndex.foreach{case (value, idx) =>
      assert(value == (idx+1) * rows)
    }
  }
}
