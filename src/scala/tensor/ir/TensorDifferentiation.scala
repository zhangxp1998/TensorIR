package tensor.ir

import lms.core.Backend._
import lms.core._
import lms.core.stub.Adapter.typeMap
import lms.core.stub._

import scala.language.implicitConversions
import scala.util.continuations._
import lms.macros.SourceContext

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
      val outputSize = x.getConv2dOutputSize(d.dims(1), that.x.dims(1), that.x.dims(2), padding, stride)
      val y = new TensorR(x.conv2d(that.x, bias.x, padding, stride), Tensor.zero[A](outputSize, AllocationType.Intermediate))
      k(y)
      val gradients = Unwrap(that.d.data)
      val kernels = Unwrap(that.x.data)
      Adapter.g.reflectEffect(
        "conv2d-backprop", Unwrap(x.data), Unwrap(y.x.data), Unwrap(d.data), Unwrap(y.d.data), Backend.Const(Seq(padding, stride)), gradients
      )(
        Unwrap(x.data), Unwrap(y.x.data), kernels
      )(
        Unwrap(d.data), gradients
      )
    }
    def relu(): TensorR[A]@diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR(x.relu(), d.copy())
      k(y)
      d.mapInplaceWithFlatIdx(i => __ifThenElse(x.unsafe_apply(i) <= 0.asInstanceOf[A], 0.asInstanceOf[A], 1.asInstanceOf[A]))
    }

    def batchNorm(gamma: TensorR[A], beta: TensorR[A], recomp: Boolean = false): TensorR[A]@diff = shift {k: (TensorR[A] => Unit) =>
      val cache = x.batchNorm(gamma.x, beta.x)
      val y = new TensorR(cache._1, Tensor.zero[A](cache._1.dims, AllocationType.Gradient))
      k(y)
      val (outy, xhat, saveMean, saveInvVariance) = if (recomp) x.batchNorm(gamma.x, beta.x) else cache
      Adapter.g.reflectEffect(
        "batchNorm-backprop", Seq(x, xhat, saveMean, saveInvVariance, gamma.x, beta.d, d, gamma.d, beta.d).map(a => Unwrap(a.data)): _*
      )(
        Seq(x, xhat, saveMean, saveInvVariance, gamma.x, beta.x).map(a => Unwrap(a.data)): _*
      )(
        Seq(d, gamma.d, beta.d).map(a => Unwrap(a.data)): _*
      )
    }
    def flatten(): TensorR[A]@diff = shift { k: (TensorR[A] => Unit) =>
      val y = new TensorR(x.flatten(), Tensor.zero[A](d.dims, AllocationType.Gradient))
      k(y)
      d = y.d
    }
  }
}

trait TensorDifferentiationCodegen extends BaseGenTensorOps {
  registerTopLevelFunction("matmul_backprop") {
    emit(
      """
        |void matmul_backprop(const float *m1, const float *m2, const float *y,
        |     float *d1, float *d2, const size_t M, const size_t K, const size_t N) {
        |  // m1: M*K, m2: K*N, y: M*N
        |  // d1 += y * m2.T => M*N x N*K = M*K
        |  // d2 += m1.T * y => K*M x M*N = K*N
        |  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, y, M, m2, K, 1.0f, d1, M);
        |  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, M, N, 1.0f, m1, M, y, M, 1.0f, d2, M);
        |}
        |""".stripMargin)
  }
  override def shallow(n: Node): Unit = n match {
    case Node(s, "matmul-backprop", List(m1, m2, y, d1, d2, Backend.Const(Seq(m: Int, k: Int, n: Int))), _) =>
      emit("matmul_backprop(")
      shallow(m1)
      emit(", ")
      shallow(m2)
      emit(", ")
      shallow(y)
      emit(", ")
      shallow(d1)
      emit(", ")
      shallow(d2)
      emit(s", $m, $k, $n)")
    case Node(s, "conv-backprop", List(x, kernel, output, d, kernelD, outputD, Backend.Const(Seq(padding: Int, stride: Int))), _) =>
      // TODO implement convolution backprop
      emitStubComment(n.op)
    case Node(s, "batchNorm-backprop", List(x, xhat, saveMean, saveInvVariance, gamma, beta, d, gamma_d, beta_d), _) =>
    // TODO implement batchnorm backprop
      emitStubComment(n.op)
    case Node(s, "conv2d-backprop", x::y_x::d::y_d::Backend.Const(Seq(padding: Int, stride: Int)):: gradients, _)=>
    // TODO implement conv2d backprop
      emitStubComment(n.op)
    case _ => super.shallow(n)
  }
  def emitStubComment(op: String): Unit = emit(s"/*${op}*/")
}

abstract class TensorDiffDriverC[A: Manifest, B: Manifest] extends TensorDriverC[A, B] with TensorDifferentiation { q =>
  override val codegen = new TensorDifferentiationCodegen {
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