package tensor.ir

import org.scalatest.FunSuite
import lms.macros.SourceContext

class CPUTensorDiffTest extends FunSuite {
  test("add") {
    val dslDriver = new CPUTensorDiffDriverC[String,Unit] {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        val length = 20
        val x = Tensor[Float](Seq(length), AllocationType.Data)
        x.mapInplaceWithFlatIdx(idx => idx)
        val gradient = TensorR.grad(a => a+a)(x)
        for (i <- 0 until length : Rep[Range]) {
          println(gradient.data(i))
        }
      }
    }

    val res = dslDriver.eval("0")
    assert(res.map(_.toDouble).forall(_ == 2))
  }

  test("sub") {
    val dslDriver = new CPUTensorDiffDriverC[String,Unit] {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        val length = 20
        val x = Tensor[Float](Seq(length), AllocationType.Data)
        x.mapInplaceWithFlatIdx(idx => idx)
        val gradient = TensorR.grad(a => TensorR[Float](Seq(length), 0)-a)(x)
        for (i <- 0 until length : Rep[Range]) {
          println(gradient.data(i))
        }
      }
    }

    val res = dslDriver.eval("5")
    assert(res.map(_.toDouble).forall(_ == -1))
  }

  test("mul") {
    val length = 20
    val dslDriver = new CPUTensorDiffDriverC[String,Unit] {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor[Float](Seq(length), AllocationType.Data)
        x.mapInplaceWithFlatIdx(idx => idx)
        val y = Tensor[Float](Seq(length), AllocationType.Data)
        y.mapInplaceWithFlatIdx(idx => 20-idx)
        val (gradient1, gradient2) = TensorR.grad((a, b) => a*b)(x, y)
        for (i <- 0 until length : Rep[Range]) {
          printf("%f %f\n", gradient1.data(i), gradient2.data(i))
        }
      }
    }

    val res = dslDriver.eval("5")
    assert(res.zipWithIndex
      .map{case (str, idx) => (str.split(" +"), idx)}
      .forall{case (arr, idx) => arr.map(_.toDouble).toSeq == Seq(length-idx.toDouble, idx.toDouble)}
    )
  }

  test("div") {
    val length = 20
    val dslDriver = new CPUTensorDiffDriverC[String,Unit] {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor[Float](Seq(length), AllocationType.Data)
        x.mapInplaceWithFlatIdx(idx => idx)
        val y = Tensor[Float](Seq(length), AllocationType.Data)
        y.mapInplaceWithFlatIdx(idx => 20-idx)
        val (gradient1, gradient2) = TensorR.grad((a, b) => a/b)(x, y)
        for (i <- 0 until length : Rep[Range]) {
          printf("%f %f\n", gradient1.data(i), gradient2.data(i))
        }
      }
    }

    val eq = (a: Double, b: Double) => Math.abs(a-b) <= 1e-6

    val res = dslDriver.eval("5")
    res.zipWithIndex
      .map{case (str, idx) => (str.split(" +").map(_.toDouble), idx)}
      .foreach{case (arr, idx) =>
        assert(eq(arr.head, 1.0f/(20-idx)))
        assert(eq(arr.tail.head, -idx/Math.pow(20-idx, 2).toFloat))
      }
  }

  test("composite") {
    val dslDriver = new CPUTensorDiffDriverC[String,Unit] {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        val length = 20
        val x = Tensor[Float](Seq(length), AllocationType.Data)
        x.mapInplaceWithFlatIdx(idx => idx)
        val gradient = TensorR.grad(a => a*2 + a*a*a)(x)
        for (i <- 0 until length : Rep[Range]) {
          println(gradient.data(i))
        }
      }
    }

    val res = dslDriver.eval("0")
    res.map(_.toDouble).zipWithIndex.foreach{
      case (d, i) => assert(d == 2 + 3*i*i)
    }
  }
}
