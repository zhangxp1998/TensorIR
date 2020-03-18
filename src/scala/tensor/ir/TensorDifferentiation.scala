package tensor.ir

import lms.core.Backend._
import lms.core._
import lms.core.stub.Adapter.typeMap
import lms.core.stub._

import scala.language.implicitConversions
import scala.util.continuations._
import lms.macros.SourceContext

import scala.tensor.ir.backend.CPUDiffTensorCodeGen

trait Diff {
  type diff = cps[Unit]
}

trait TensorDifferentiation extends TensorOps {

  object TensorR {
    def apply[T: Manifest: Numeric](dims: Seq[Int], fillVal: T): TensorR[T] = {
      val tensor = Tensor.fill[T](dims, fillVal, AllocationType.Intermediate)
      TensorR(tensor)
    }
    def apply[T: Manifest: Numeric](x: Tensor[T]): TensorR[T] = {
      new TensorR(x, Tensor.zero[T](x.dims, AllocationType.Gradient))
    }
    def apply[T: Manifest: Numeric](dims: Seq[Int], f: Rep[Int] => T): TensorR[T] = {
      val tensor = Tensor[T](dims, AllocationType.Intermediate)
      tensor.mapInplaceWithFlatIdx(idx => f(idx))
      TensorR(tensor)
    }
    def rand(dims: Seq[Int], allocType: AllocationType): TensorR[Float] = {
      TensorR(Tensor.rand(dims, allocType))
    }
    def grad(f: TensorR[Float] => TensorR[Float]@cps[Unit])(x: Tensor[Float]): Tensor[Float] = {
      val z = new TensorR[Float](x, Tensor.zero[Float](x.dims, AllocationType.Gradient))
      reset({
        val res = f(z)
        res.d = Tensor.fill[Float](res.x.dims, 1, AllocationType.Gradient)
      })
      z.d
    }

    def grad(f: (TensorR[Float], TensorR[Float]) => TensorR[Float]@cps[Unit])
            (x: Tensor[Float], y: Tensor[Float]): (Tensor[Float], Tensor[Float]) = {
      val z1 = new TensorR[Float](x, Tensor.zero[Float](x.dims, AllocationType.Gradient))
      val z2 = new TensorR[Float](y, Tensor.zero[Float](x.dims, AllocationType.Gradient))
      reset({
        val res = f(z1, z2)
        res.d = Tensor.fill[Float](res.x.dims, 1, AllocationType.Gradient)
      })
      (z1.d, z2.d)
    }
  }

  class TensorR[A: Manifest : Numeric](recompute: => Tensor[A], var d: Tensor[A], val shouldRecompute: Boolean = false) extends Diff {
    // Single element broadcast operations
    lazy val cache: Tensor[A] = recompute
    def x: Tensor[A] = if (shouldRecompute) {
      recompute
    }else {
      cache
    }

    def matmul(that: TensorR[A]): TensorR[A]@diff = shift { k: (TensorR[A] => Unit) =>
      val M = cache.dims.head
      val K = that.cache.dims.head
      val N = that.cache.dims(1)
      val y = new TensorR((cache.matmul(that.cache, AllocationType.Intermediate)): Tensor[A], Tensor.zero(Seq(M, N), AllocationType.Gradient))
      k(y)
      val m1: Tensor[A] = x
      val m2: Tensor[A] = that.x
      val output: Tensor[A] = y.x

      Adapter.g.reflectEffect(
        "matmul-backprop", Unwrap(m1.data), Unwrap(m2.data), Unwrap(output.data), Unwrap(d.data), Unwrap(that.d.data), Backend.Const(Seq(M, K, N))
      )(
        Unwrap(m1.data), Unwrap(m2.data), Unwrap(output.data)
      )(
        Unwrap(d.data), Unwrap(that.d.data)
      )
    }

    def +(that: Rep[A]): TensorR[A]@diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR((x + that): Tensor[A], Tensor.zero[A](x.dims, AllocationType.Gradient))
      k(y)
      this.d += y.d
    }

    def -(that: Rep[A]): TensorR[A]@diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR((x - that): Tensor[A], Tensor.zero[A](x.dims, AllocationType.Gradient))
      k(y)
      this.d += y.d
    }

    def *(that: Rep[A]): TensorR[A]@diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR(x * that, Tensor.zero[A](x.dims, AllocationType.Gradient))
      k(y)
      this.d += y.d * that
    }

    def /(that: Rep[A]): TensorR[A]@diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR(x / that, Tensor.zero[A](x.dims, AllocationType.Gradient))
      k(y)
      this.d += y.d / that
    }

    // Tensor-Tensor element wise operations
    def +(that: TensorR[A]): TensorR[A]@diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR(x add that.x, Tensor.zero[A](that.x.dims, AllocationType.Gradient))
      k(y)
      that.d += y.d
      this.d += y.d
    }

    def -(that: TensorR[A]): TensorR[A]@diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR(x sub that.x, Tensor.zero[A](that.x.dims, AllocationType.Gradient))
      k(y)
      that.d -= y.d
      this.d += y.d
    }

    def *(that: TensorR[A]): TensorR[A]@diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR(x mul that.x, Tensor.zero[A](that.x.dims, AllocationType.Gradient))
      k(y)
      that.d += y.d mul this.x
      this.d += y.d mul that.x
    }

    def /(that: TensorR[A]): TensorR[A]@diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR(x mul that.x, Tensor.zero[A](that.x.dims, AllocationType.Gradient))
      k(y)
      that.d -= y.d mul this.x div (that.x mul that.x)
      this.d += y.d div that.x
    }
    def conv(that: TensorR[A], padding: Int, stride: Int): TensorR[A]@diff = shift {k: (TensorR[A] => Unit) =>
      val outputSize = x.getConvOutputSize(that.x.dims, padding, stride)
      val y = new TensorR(x.conv(that.x, padding, stride), Tensor.zero[A](outputSize, AllocationType.Gradient))
      k(y)
      Adapter.g.reflectEffect(
        "conv-backprop", Unwrap(x.data), Unwrap(that.x.data), Unwrap(y.x.data), Unwrap(d.data), Unwrap(that.d.data), Unwrap(y.d.data), Backend.Const(Seq(padding, stride))
      )(
        Unwrap(x.data), Unwrap(y.x.data), Unwrap(that.x.data)
      )(
        Unwrap(d.data), Unwrap(that.d.data)
      )
    }

    def conv2d(that: TensorR[A], bias: TensorR[A], padding: Int, stride: Int): TensorR[A]@diff = shift { k: (TensorR[A] => Unit) =>
      assert(d.dims.length == 4)
      assert(that.d.dims.length == 4)
      val Seq(oc, ic, kh, kw) = that.d.dims
      val outputSize = x.getConv2dOutputSize(ic, oc, kh, padding, stride)
      val y = new TensorR(x.conv2d(that.x, bias.x, padding, stride), Tensor.zero[A](outputSize, AllocationType.Intermediate))
      k(y)
      val gradients = Unwrap(that.d.data)
      val kernels = Unwrap(that.x.data)
      Adapter.g.reflectEffect(
        "conv2d-backprop", Backend.Const(d.dims) +: Backend.Const(Seq(oc, kh, padding, stride)) +: Seq(y.d, x, that.d, bias.d).map(a => Unwrap(a.memDesc)): _*
      )(
        Unwrap(y.d.data), Unwrap(x.data)
      )(
        Unwrap(that.d.data), Unwrap(bias.d.data)
      )
    }
    def relu(): TensorR[A]@diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR(x.relu(), d.copy())
      k(y)
      d.mapInplaceWithFlatIdx(i => __ifThenElse(x.unsafe_apply(i) <= 0.asInstanceOf[A], 0.asInstanceOf[A], 1.asInstanceOf[A]))
    }

    def batchNorm(gamma_beta: TensorR[A], recomp: Boolean = false): TensorR[A]@diff = shift {k: (TensorR[A] => Unit) =>
      val cache = x.batchNorm(gamma_beta.x)
      val y = new TensorR(cache._1, Tensor.zero[A](cache._1.dims, AllocationType.Gradient))
      k(y)
      val (_, avg, variance) = if (recomp) x.batchNorm(gamma_beta.x) else cache
      Adapter.g.reflectEffect(
        "batchNorm-backprop", Backend.Const(d.dims)+:Seq(x, y.d, avg, variance, d, gamma_beta.x, gamma_beta.d).map(a => Unwrap(a.memDesc)): _*
      )(
        Seq(x, y.d, avg, variance, gamma_beta.x).map(a => Unwrap(a.data)): _*
      )(
        Unwrap(d, gamma_beta.d.data)
      )
    }
    def flatten(): TensorR[A]@diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR(x.flatten(), Tensor.zero[A](d.dims, AllocationType.Gradient))
      k(y)
      d = y.d
    }
    def sum(): TensorR[A]@diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR(x.sumT(), Tensor.zero[A](Seq(1), AllocationType.Gradient))
      k(y)
      val gradient = y.d.unsafe_apply(0)
      d.transformRange(0, d.dims.product, _ => gradient)
    }
  }
}

abstract class TensorDiffDriverC[A: Manifest, B: Manifest] extends TensorDriverC[A, B] with TensorDifferentiation { q =>
  override val codegen = new CPUDiffTensorCodeGen {
    override val IR: q.type = q
  }
}

object NumR {
  implicit def toNumR(x: Double): NumR = new NumR(x, 0)

  def apply(x: Double): NumR = toNumR(x)

  def grad(f: NumR => NumR@cps[Unit])(x: Double): Double = {
    val z = new NumR(x, 0.0)
    reset {
      f(z).d = 1.0
    }
    z.d
  }

  def grad(f: (NumR, NumR) => NumR@cps[Unit])(x: Double, y: Double): (Double, Double) = {
    val z1 = new NumR(x, 0.0)
    val z2 = new NumR(y, 0.0)
    reset {
      f(z1, z2).d = 1.0
    }
    (z1.d, z2.d)
  }
}

class NumR(val x: Double, var d: Double) extends Diff {

  def +(that: NumR): NumR@diff = shift { k: (NumR => Unit) =>
    val y = new NumR(x + that.x, 0)
    k(y)
    that.d += y.d
    this.d += y.d
  }

  def -(that: NumR): NumR@diff = shift { k: (NumR => Unit) =>
    val y = new NumR(x - that.x, 0)
    k(y)
    that.d -= y.d
    this.d += y.d
  }

  def *(that: NumR): NumR@diff = shift { k: (NumR => Unit) =>
    val y = new NumR(x * that.x, 0)
    k(y)
    that.d += y.d * this.x
    this.d += y.d * that.x
  }

  def /(that: NumR): NumR@diff = shift { k: (NumR => Unit) =>
    val y = new NumR(x / that.x, 0)
    k(y)
    that.d -= y.d * this.x / (that.x * that.x)
    this.d += y.d / that.x
  }
}


object TensorDifferentiation {
  def main(args: Array[String]): Unit = {
    val dslDriver = new TensorDiffDriverC[String, Unit] {
      override def snippet(x: Rep[String]): Rep[Unit] = {

        def grad(f: TensorR[Float] => TensorR[Float]@cps[Unit])(x: Tensor[Float]): Tensor[Float] = {
          val z = new TensorR[Float](x, Tensor.zero[Float](x.dims, AllocationType.Gradient))
          reset({
            val res = f(z)
            res.d = Tensor.fill[Float](res.x.dims, 1, AllocationType.Gradient)
            val a: Tensor[Float] = res.x
            println(a(0))
            println(a(1))
          })
          z.d
        }

        val x = Tensor.fill[Float](Seq(2), 2, AllocationType.Data)
        val gradient = grad(a => a * a * a * a)(x)
        println(gradient(0))
        println(gradient(1))
      }
    }


    val res = dslDriver.eval("5")
    println(res.mkString("\n"))
  }
}