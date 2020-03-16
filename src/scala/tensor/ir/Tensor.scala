package tensor.ir

import java.io.PrintWriter

import lms.core.Backend.{Const, _}
import lms.core._
import lms.core.stub._
import lms.macros.SourceContext
import scala.tensor.ir.backend.CPUTensorCodeGen

object AllocationType extends Enumeration {
  type AllocationType = Value
  val Data, Gradient, Intermediate, Parameter = Value
}

trait TensorOps extends Base with Equal with OrderingOps with PrimitiveOps with RandomOps {
  type AllocationType = AllocationType.AllocationType
  abstract class DataLoop {
    def foreach(f: Rep[Int] => Unit): Unit
  }

  object DataLoop {
    def apply(size: Int) = if (size <= 3) {
      new DataLoop {
        def foreach(f: Rep[Int] => Unit) = {
          for (i <- 0 until size: Range) f(unit(i))
        }
      }
    } else {
      new DataLoop {
        def foreach(f: Rep[Int] => Unit) = {
          for (i <- 0 until size: Rep[Range]) f(i)
        }
      }
    }

    def apply(size: Rep[Int]) = {
      new DataLoop {
        def foreach(f: Rep[Int] => Unit) = {
          for (i <- 0 until size) f(i)
        }
      }
    }
  }
  // A trait that maps to dnnl::memory::dims. It holds the dimension of tensors at runtime
  trait MemDims {
  }
  // A trait that maps to dnnl::memory. It holds the data pointer, format, and memory dims at runtime.
  trait MemDesc {
  }
  object Tensor {
    def apply[A: Manifest: Numeric](xs: Seq[Int], allocType: AllocationType)(implicit pos: SourceContext): Tensor[A] = {
      new Tensor(xs, allocType)
    }
    def fill[A: Manifest: Numeric](dims: Seq[Int], fillVal: A, allocType: AllocationType)(implicit pos: SourceContext): Tensor[A] = {
      val tensor = Tensor[A](dims, allocType)
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(tensor.data), Unwrap(fillVal), Backend.Const(dims))
      Wrap[Unit](Adapter.g.reflectWrite("tensor-fill", unwrapped_xs:_*)(Unwrap(tensor.data)))
      tensor
    }
    def zero[A: Manifest: Numeric](dims: Seq[Int], allocType: AllocationType)(implicit pos: SourceContext): Tensor[A] = {
      Tensor.fill[A](dims, 0.asInstanceOf[A], allocType)
    }
    def rand(dims: Seq[Int], allocType: AllocationType)(implicit pos: SourceContext): Tensor[Float] = {
      val tensor = Tensor[Float](dims, allocType)
      tensor.mapInplace(_ => randFloat())
      tensor
    }
    def createMemDims(dims: Seq[Int]): Rep[MemDims] = {
      Wrap[MemDims](Adapter.g.reflect("mem-dims", Backend.Const(dims)))
    }
    def createMemDesc[A: Manifest: Numeric](memDims: Rep[MemDims], tensor: Tensor[A]): Rep[MemDesc] = {
      Wrap[MemDesc](Adapter.g.reflect("mem-desc", Unwrap(memDims), Unwrap(tensor.data), Backend.Const(tensor.dims)))
    }
  }
  class Tensor[A: Manifest: Numeric] (val dims: Seq[Int], var data: Rep[Array[A]], val allocType: AllocationType) {
    lazy val memDims: Rep[MemDims] = Tensor.createMemDims(dims)
    lazy val memDesc: Rep[MemDesc] = Tensor.createMemDesc(memDims, this)
    def infix_+(a: Rep[A], b: Rep[A]): Rep[A] = Wrap[A](Adapter.g.reflect("+", Unwrap(a), Unwrap(b)))
    def infix_-(a: Rep[A], b: Rep[A]): Rep[A] = Wrap[A](Adapter.g.reflect("-", Unwrap(a), Unwrap(b)))
    def infix_*(a: Rep[A], b: Rep[A]): Rep[A] = Wrap[A](Adapter.g.reflect("*", Unwrap(a), Unwrap(b)))
    def infix_/(a: Rep[A], b: Rep[A]): Rep[A] = Wrap[A](Adapter.g.reflect("/", Unwrap(a), Unwrap(b)))
    def sqrt(a: Rep[A]): Rep[A] = Wrap[A](Adapter.g.reflect("sqrt", Unwrap(a)))

    lazy val strides = dims.scanRight(1)(_ * _).tail
    def this(dims: Seq[Int], allocType: AllocationType) {
      this(dims, null, allocType)
      data = {
        val mA = Backend.Const(manifest[A])
        Wrap[Array[A]](Adapter.g.reflectMutable("tensor-new", mA, Backend.Const(dims), Backend.Const(allocType)))
      }
    }
    def checkIdx(idx: Seq[Int]): Unit = {
      assert(dims.length == idx.length, s"Tensor index $idx does not match dimension $dims")
      assert(idx.zip(dims).forall{case (a, b) => a < b}, s"Tensor index $idx is out of bounds for dimension $dims")
    }
    def apply(idx: Int*): Rep[A] = {
      checkIdx(idx)
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(data), Backend.Const(idx), Backend.Const(dims))
      Wrap[A](Adapter.g.reflectRead("tensor-apply", unwrapped_xs:_*)(Unwrap(data)))
    }
    def update(idx: Seq[Int], newVal: A): Unit = {
      update(idx, Const(newVal))
    }

    def update(idx: Seq[Int], newVal: Rep[A]): Unit = {
      checkIdx(idx)
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(data), Backend.Const(idx), Unwrap(newVal), Backend.Const(dims))
      Wrap[Unit](Adapter.g.reflectWrite("tensor-update", unwrapped_xs:_*)(Unwrap(data)))
    }
    def unsafe_apply(idx: Rep[Int]): Rep[A] = {
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(data), Unwrap(idx))
      Wrap[A](Adapter.g.reflectRead("tensor-apply", unwrapped_xs:_*)(Unwrap(data)))
    }

    private def unsafe_update(idx: Rep[Int], newVal: Rep[A]): Unit = {
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(data), Unwrap(idx), Unwrap(newVal))
      Wrap[Unit](Adapter.g.reflectWrite("tensor-update", unwrapped_xs:_*)(Unwrap(data)))
    }
    def mapInplace(f: Rep[A] => Rep[A]): Unit = {
      transformRange(0, dims.product, f)
    }
    def mapInplaceWithFlatIdx(f: Rep[Int] => Rep[A]): Unit = {
      val mA = Backend.Const(manifest[A])
      val block = Adapter.g.reify(exp => Unwrap(f(Wrap[Int](exp))))
      Wrap[Unit](Adapter.g.reflectEffect(
        "tensor-transform-index", mA, Unwrap(data), block, Backend.Const(dims)
      )(
        (block.eff.rkeys + Unwrap(data)).toSeq: _*
      )(
        (block.eff.wkeys + Unwrap(data)).toSeq: _*
      ))
    }
    def copy(newAllocType: AllocationType): Tensor[A] = {
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(data), Backend.Const(dims), Backend.Const(newAllocType))
      new Tensor(
        dims,
        Wrap[Array[A]](Adapter.g.reflectRead("tensor-copy", unwrapped_xs:_*)(Unwrap(data), STORE)),
        newAllocType
      )
    }

    def copy(): Tensor[A] = copy(allocType)

    private def binary_broadcast(op: (Rep[A], Rep[A]) => Rep[A], rhs: Rep[A]): Tensor[A] = {
      val result: Tensor[A] = copy()
      result.mapInplace(op(_, rhs))
      result
    }

    def checkDims(rhs_dims: Seq[Int]): Unit = {
      if (rhs_dims != dims) {
        throw new RuntimeException(s"$rhs_dims is not the same as $dims")
      }
    }
    private def tensor_binary(rhs: Tensor[A], op: String): Tensor[A] = {
      checkDims(rhs.dims)
      val res = copy()
      res.tensor_binary_inplace(rhs, op)
      res
    }
    private def tensor_binary_inplace(rhs: Tensor[A], op: String): Unit = {
      mapInplaceWithFlatIdx(idx => {
        val a: Rep[A] = unsafe_apply(idx)
        val b: Rep[A] = rhs.unsafe_apply(idx)
        val c: Rep[A] = Wrap[A](Adapter.g.reflect(op, Unwrap(a), Unwrap(b)))
        c
      }
      )
    }

    def add(rhs: Tensor[A]): Tensor[A] = {
      tensor_binary(rhs, "+")
    }
    def sub(rhs: Tensor[A]): Tensor[A] = {
      tensor_binary(rhs, "-")
    }
    def mul(rhs: Tensor[A]): Tensor[A] = {
      tensor_binary(rhs, "*")
    }
    def div(rhs: Tensor[A]): Tensor[A] = {
      tensor_binary(rhs, "/")
    }

    def +=(rhs: Tensor[A]): Unit = {
      tensor_binary_inplace(rhs, "+")
    }
    def -=(rhs: Tensor[A]): Unit = {
      tensor_binary_inplace(rhs, "-")
    }
    def *=(rhs: Tensor[A]): Unit = {
      tensor_binary_inplace(rhs, "*")
    }
    def /=(rhs: Tensor[A]): Unit = {
      tensor_binary_inplace(rhs, "/")
    }

    def +(rhs: Rep[A]): Tensor[A] = {
      binary_broadcast((a, b) => Wrap[A](Adapter.INT(Unwrap(a)).+(Adapter.INT(Unwrap(b))).x): Rep[A], rhs)
    }
    def -(rhs: Rep[A]): Tensor[A] = {
      binary_broadcast((a, b) => Wrap[A](Adapter.INT(Unwrap(a)).-(Adapter.INT(Unwrap(b))).x): Rep[A], rhs)
    }
    def *(rhs: Rep[A]): Tensor[A] = {
      binary_broadcast((a, b) => Wrap[A](Adapter.INT(Unwrap(a)).*(Adapter.INT(Unwrap(b))).x): Rep[A], rhs)
    }
    def /(rhs: Rep[A]): Tensor[A] = {
      binary_broadcast((a, b) => Wrap[A](Adapter.INT(Unwrap(a))./(Adapter.INT(Unwrap(b))).x): Rep[A], rhs)
    }

    def matmul(rhs: Tensor[A], allocType: AllocationType): Tensor[A] = {
      val rhs_dims: Seq[Int] = rhs.dims
      val lhs_dims: Seq[Int] = dims
      val mA = Backend.Const(manifest[A])
      if (lhs_dims.length != 2) {
        throw new RuntimeException(s"$lhs_dims must be 2D")
      }
      if (lhs_dims(1) != rhs_dims.head) {
        throw new RuntimeException(s"$lhs_dims and $rhs_dims are not compatible")
      }
      val M: Int = lhs_dims.head
      val K: Int = rhs_dims.head
      val N: Int = rhs_dims match {
        case _::tail::Nil => tail
        case _::Nil => 1
      }

      // vector-vector multiplication
      val result = Tensor[A](Seq(M, N), allocType)
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(data), Unwrap(rhs.data), Unwrap(result.data), Backend.Const(Seq(M, K, N)))
      Wrap[Unit](Adapter.g.reflectEffect("matrix-multiply", unwrapped_xs:_*)(Unwrap(data), Unwrap(rhs.data))(Unwrap(result.data)))
      result
    }
    def getConvOutputSize(kernelSize: Seq[Int], pading: Int, stride: Int): Seq[Int] = {
      val (left, right) = dims.splitAt(dims.length - kernelSize.length)
      val outputRight = right.zip(kernelSize).map{case (input, kernel) => (input+pading*2 - (kernel- 1))/stride}
      val outputDims = left ++ outputRight
      outputDims
    }
    def getConv2dOutputSize(inChannels: Int, outChannels: Int, kernelSize: Int, padding: Int, stride: Int): Seq[Int] = {
      Seq[Int](dims.head, outChannels) ++ getConvOutputSize(Seq(kernelSize, kernelSize), padding, stride).tail.tail
    }
    def conv(rhs: Tensor[A], pading: Int, stride: Int): Tensor[A] = {
      if (dims.length < 2) {
        throw new IllegalAccessError("Convolution can only be done on 3d or 4d tensors")
      }
      if (rhs.dims.length > dims.length) {
        throw new IllegalArgumentException(s"Kernel cannot be bigger than input, kernel dims: ${rhs.dims} input dims: $dims")
      }
      val outputDims = getConvOutputSize(rhs.dims, pading, stride)
      val output = Tensor[A](outputDims, AllocationType.Intermediate)
      val mA = Backend.Const(manifest[A])
      Wrap[Unit](Adapter.g.reflectEffect(
        "tensor-convolution",
        mA, Unwrap(data), Unwrap(rhs.data), Unwrap(output.data), Backend.Const(dims), Backend.Const(rhs.dims)
      )(Unwrap(data), Unwrap(rhs.data))(Unwrap(output.data)))
      output
    }
    def conv2d(rhs: Tensor[A], bias: Tensor[A], padding: Int, stride: Int): Tensor[A] = {
      val Seq(ic, oc, kh, kw) = rhs.dims
      assert(dims.length == 4, s"Convolution can only be done on 4d tensors $dims")
      assert(rhs.dims.length == 4, s"Kernel must have dimmension of 4 $rhs")
      assert(ic == dims(1), s"Kernel input channel should match data ${rhs.dims} $dims")
      assert(kh == kw, s"Kernel should be a square $kh, $kw")

      val mA = Backend.Const(manifest[A])
      val outputSize = getConv2dOutputSize(dims(1), oc, kh, padding, stride)
      val output = Tensor[A](outputSize, AllocationType.Intermediate)
      val kernelsUnwrapped = Unwrap(rhs.data)
      Wrap[Unit](Adapter.g.reflectEffect(
        "tensor-convolution2d",
        mA, Unwrap(memDesc), Unwrap(output.memDesc), Unwrap(rhs.memDesc), Unwrap(bias.memDesc), Backend.Const(dims), Backend.Const(Seq(oc, kh, padding, stride))
      )(
        Unwrap(data), kernelsUnwrapped
      )(
        Unwrap(output.data)
      )
      )
      output
    }

    def relu(inplace: Boolean = false): Tensor[A] = {
      val output = if (inplace) this else copy()
      output.mapInplace(ordering_max(_, 0.asInstanceOf[A]))
      output
    }
    def dropout(p: Float = 0.5, inplace: Boolean = false): Tensor[A] = {
      assert(0.0f <= p && p < 1.0f, s"dropout rate should be [0.0, 1), got $p")
      val output = if (inplace) this else copy()
      output.mapInplace(a => __ifThenElse(randFloat() < p, Wrap[A](Adapter.g.reflect("/", Unwrap(a), Backend.Const(p))), 0.asInstanceOf[A]))
      output
    }
    def accumulateRange(begin: Rep[Int], end: Rep[Int]): Rep[A] = {
      val mA = Backend.Const(manifest[A])
      Wrap[A](Adapter.g.reflectRead("tensor-accumulate-range", mA, Unwrap(data), Unwrap(begin), Unwrap(end))(Unwrap(data)))
    }
    def sum(): Rep[A] = accumulateRange(0, dims.product)
    def square(): Tensor[A] = {
      val output = copy()
      output.mapInplace(a => infix_*(a, a))
      output
    }
    def sqrt(): Tensor[A] = {
      val output = copy()
      output.mapInplace(sqrt)
      output
    }
    def transformRange(begin: Rep[Int], end: Rep[Int], f: Rep[A] => Rep[A]): Unit = {
      val mA = Backend.Const(manifest[A])
      val block = Adapter.g.reify(exp => Unwrap(f(Wrap[A](exp))))
      Wrap[Unit](Adapter.g.reflectEffect(
        "tensor-transform-range", mA, Unwrap(data), block, Unwrap(begin), Unwrap(end)
      )(
        (block.eff.rkeys + Unwrap(data)).toSeq: _*
      )(
        (block.eff.wkeys + Unwrap(data)).toSeq: _*
      ))
    }
    def batchNorm(gamma_beta: Tensor[A]) = {
      assert(dims.length == 4, "BatchNorm only supports 4d tensor")
      val Seq(n, c, h, w) = dims
      assert(gamma_beta.dims == Seq(2, c), s"Beta and Gamma should have dims ${Seq(2, c)}")
      val dst = Tensor[A](dims, AllocationType.Intermediate)
      val epsilon = 0.00001f
      val avg = Tensor[A](Seq(c), AllocationType.Intermediate)
      val variance = Tensor[A](Seq(c), AllocationType.Intermediate)
      Wrap[Unit](
        Adapter.g.reflectEffect(
          "batchnorm-forward", Backend.Const(dims)+:Backend.Const(epsilon)+:Seq(this, avg, variance, gamma_beta, dst).map(a => Unwrap(a.memDesc)): _*
        )(
          Unwrap(data), Unwrap(gamma_beta.data)
        )(
          Seq(dst, avg, variance).map(a => Unwrap(a.data)): _*
        )
      )
      (dst, avg, variance)
    }
    def flatten(): Tensor[A] = {
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(data), Backend.Const(dims))
      new Tensor(
        Seq(dims.product),
        Wrap[Array[A]](Adapter.g.reflectRead("tensor-copy", unwrapped_xs:_*)(Unwrap(data), STORE)),
        allocType
      )
    }
  }
  def println(x: Tensor[_]): Unit = {
    println(x.data)
  }
}

abstract class TensorDriverC[A: Manifest, B: Manifest] extends DslDriverC[A, B] with TensorOps { q =>
  override val codegen = new CPUTensorCodeGen {
    override val IR: q.type = q
  }
  lazy val g: Graph = Adapter.program(Adapter.g.reify(x => Unwrap(wrapper(Wrap[A](x)))))
}

object Runer {

  def main(args: Array[String]) {
    val dslDriver = new TensorDriverC[String,Unit] {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        var tensor = Tensor.fill[Float](Seq(1, 2, 3), 4.0, AllocationType.Data)
        tensor = tensor + 1
        tensor = tensor - 1
        tensor = tensor + 1
        tensor = tensor - 1
        val tensor2 = Tensor[Float](Seq(1, 2, 3), AllocationType.Data)
        tensor2.mapInplaceWithFlatIdx(idx => idx)
        println(tensor)

        println(tensor(0, 1, 2))
        println(tensor.copy()(0, 1, 2))
        println((tensor add tensor)(0, 0, 0))
        println((tensor+tensor(0, 0, 0))(0, 1, 2))
        println(tensor2(0, 1, 2))

        val mat1 = Tensor.fill[Float]((Seq(3, 3)), 0, AllocationType.Data)
        mat1(Seq(0, 0)) = 1.0
        mat1(Seq(1, 1)) = 1.0
        mat1(Seq(2, 2)) = 1.0
        val mat2 = Tensor[Float](Seq(3, 3), AllocationType.Data)
        mat2.mapInplaceWithFlatIdx(idx => idx)
        val mat3 = mat1.matmul(mat2, AllocationType.Data)
        println(mat3(0, 0))
        println(mat3(0, 1))
        println(mat3(0, 2))
      }
    }


    dslDriver.eval("5").foreach(println)
  }
}

