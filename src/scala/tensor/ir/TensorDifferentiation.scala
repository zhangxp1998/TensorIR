package tensor.ir

import scala.language.implicitConversions
import scala.util.continuations._
import lms.macros.SourceContext

trait Diff {
  type diff = cps[Unit]
}

trait TensorDifferentiation extends TensorOps {
  class TensorR[A: Manifest: Numeric](val x: Tensor[A], var d: Tensor[A]) extends Diff{
    // Single element broadcast operations
    def +(that: Rep[A]): TensorR[A] @diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR((x + that): Tensor[A], Tensor.zero[A](x.dims))
      k(y)
      this.d += y.d
    }
    def -(that: Rep[A]): TensorR[A] @diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR((x - that): Tensor[A], Tensor.zero[A](x.dims))
      k(y)
      this.d += y.d
    }
    def *(that: Rep[A]): TensorR[A] @diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR(x * that, Tensor.zero[A](x.dims))
      k(y)
      this.d += y.d * that
    }
    def /(that: Rep[A]): TensorR[A] @diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR(x / that, Tensor.zero[A](x.dims))
      k(y)
      this.d += y.d / that
    }

    // Tensor-Tensor element wise operations
    def +(that: TensorR[A]): TensorR[A] @diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR(x add that.x, Tensor.zero[A](x.dims))
      k(y)
      that.d += y.d
      this.d += y.d
    }
    def -(that: TensorR[A]): TensorR[A] @diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR(x sub that.x, Tensor.zero[A](x.dims))
      k(y)
      that.d -= y.d
      this.d += y.d
    }
    def *(that: TensorR[A]): TensorR[A] @diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR(x mul that.x, Tensor.zero[A](x.dims))
      k(y)
      that.d += y.d mul this.x
      this.d += y.d mul that.x
    }
    def /(that: TensorR[A]): TensorR[A] @diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR(x mul that.x, Tensor.zero[A](x.dims))
      k(y)
      that.d -= y.d mul this.x div (that.x mul that.x)
      this.d += y.d div that.x
    }
  }
}

object NumR {
  implicit def toNumR(x: Double): NumR = new NumR(x, 0)
  def apply(x: Double): NumR = toNumR(x)
  def grad(f: NumR => NumR @cps[Unit])(x: Double): Double = {
    val z = new NumR(x, 0.0)
    reset {f(z).d = 1.0}
    z.d
  }
  def grad(f: (NumR, NumR) => NumR @cps[Unit])(x: Double, y: Double): (Double, Double) = {
    val z1 = new NumR(x, 0.0)
    val z2 = new NumR(y, 0.0)
    reset {f(z1, z2).d = 1.0}
    (z1.d, z2.d)
  }
}

class NumR(val x: Double, var d: Double) extends Diff {

  def +(that: NumR): NumR @diff = shift { k: (NumR => Unit) =>
    val y = new NumR(x+that.x, 0)
    k(y)
    that.d += y.d
    this.d += y.d
  }

  def -(that: NumR): NumR @diff = shift { k: (NumR => Unit) =>
    val y = new NumR(x-that.x, 0)
    k(y)
    that.d -= y.d
    this.d += y.d
  }

  def *(that: NumR): NumR @diff = shift { k: (NumR => Unit) =>
    val y = new NumR(x*that.x, 0)
    k(y)
    that.d += y.d * this.x
    this.d += y.d * that.x
  }

  def /(that: NumR): NumR @diff = shift { k: (NumR => Unit) =>
    val y = new NumR(x/that.x, 0)
    k(y)
    that.d -= y.d * this.x/(that.x*that.x)
    this.d += y.d / that.x
  }
}




object TensorDifferentiation {
  def main(args: Array[String]): Unit = {
    val dslDriver = new TensorDriverC[String,Unit] with TensorDifferentiation {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fill[Float](Seq(2), 1)
        def grad(f: TensorR[Float] => TensorR[Float] @cps[Unit])(x: Tensor[Float]): Tensor[Float] = {
          val z = new TensorR[Float](x, Tensor.zero[Float](x.dims))
          reset({
            val res = f(z)
            res.d = Tensor.fill[Float](res.x.dims, 1)
          })
          z.d
        }
        val gradient = grad(a => a*a*a)(x)
        println(gradient(0))
        println(gradient(1))
      }
    }


    dslDriver.eval("5")
  }
}