package tensor.ir

import lms.core.Backend.{Const, _}
import lms.core._
import lms.core.stub._
import lms.macros.SourceContext
import scala.tensor.ir.backend.CPUTensorCodeGen

trait CPUTensorOps extends Printf with Equal with OrderingOps with PrimitiveOps with RandomOps {
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
  def infix_+[A: Manifest: Ordering](a: Rep[A], b: Rep[A]): Rep[A] = Wrap[A](Adapter.g.reflect("+", Unwrap(a), Unwrap(b)))
  def infix_-[A: Manifest: Ordering](a: Rep[A], b: Rep[A]): Rep[A] = Wrap[A](Adapter.g.reflect("-", Unwrap(a), Unwrap(b)))
  def infix_*[A: Manifest: Ordering](a: Rep[A], b: Rep[A]): Rep[A] = Wrap[A](Adapter.g.reflect("*", Unwrap(a), Unwrap(b)))
  def infix_/[A: Manifest: Ordering](a: Rep[A], b: Rep[A]): Rep[A] = Wrap[A](Adapter.g.reflect("/", Unwrap(a), Unwrap(b)))
  def __sqrt[A: Manifest: Ordering](a: Rep[A]): Rep[A] = Wrap[A](Adapter.g.reflect("sqrt", Unwrap(a)))
  def exp[T: Manifest: Ordering](x: Rep[T]): Rep[T] = {
    Wrap[T](Adapter.g.reflect("exp", Unwrap(x)))
  }
  def log[T: Manifest](x: Rep[T]): Rep[T] = {
    Wrap[T](Adapter.g.reflect("log", Unwrap(x)))
  }
  def __softmaxLoss[A: Manifest: Ordering](probs: Tensor[A], labels: Tensor[Int]): Rep[A] = {
    assert(labels.dims.length == 1, s"Label should be a 1D Tensor ${labels.dims}")
    val mA = Backend.Const(manifest[A])
    val (rows, rowSize) = probs.dims.length match {
      case 1 => (1, probs.dims.head)
      case _ => (probs.dims.head, probs.dims.product/probs.dims.head)
    }
    assert(labels.dims.head == rows, s"Labels should have same head dimension with data: ${rows}, ${labels.dims.head}")
    val loss = Wrap[A](Adapter.g.reflectRead(
      "tensor-nll-loss", mA, Backend.Const((rows, rowSize)), Unwrap(probs.data), Unwrap(labels.data)
    )(
      Unwrap(probs.data), Unwrap(labels.data)
    )
    )
    infix_/(loss, rows)
  }
  object Tensor {
    def apply[A: Manifest: Ordering](xs: Seq[Int], allocType: AllocationType)(implicit pos: SourceContext): Tensor[A] = {
      new Tensor(xs, allocType)
    }
    def mmap[A: Manifest: Ordering](dims: Seq[Int], path: String): Tensor[A] = {
      val mA = Backend.Const(manifest[A])
      val data = Wrap[Array[A]](Adapter.g.reflectRead("tensor-mmap", mA, Backend.Const(dims), Backend.Const(path))(Adapter.CTRL))
      new Tensor[A](dims, data, AllocationType.Data)
    }
    def fill[A: Manifest: Ordering](dims: Seq[Int], fillVal: A, allocType: AllocationType)(implicit pos: SourceContext): Tensor[A] = {
      val tensor = Tensor[A](dims, allocType)
      tensor.fill(fillVal)
      tensor
    }
    def zero[A: Manifest: Ordering](dims: Seq[Int], allocType: AllocationType)(implicit pos: SourceContext): Tensor[A] = {
      Tensor.fill[A](dims, 0.asInstanceOf[A], allocType)
    }
    def rand(dims: Seq[Int], allocType: AllocationType)(implicit pos: SourceContext): Tensor[Float] = rand(dims, 0.0f, 1.0f, allocType)
    def rand(dims: Seq[Int], lower: Float, upper: Float, allocType: AllocationType)(implicit pos: SourceContext): Tensor[Float] = {
      val tensor = Tensor[Float](dims, allocType)
      val distribution = getUniformFloatDistribution(lower, upper)
      tensor.mapInplace(_ => sampleDistribution(distribution))
      tensor
    }
    def createMemDims(dims: Seq[Int]): Rep[MemDims] = {
      Wrap[MemDims](Adapter.g.reflect("mem-dims", Backend.Const(dims)))
    }
    def copy[A: Manifest](src: Tensor[A], src_begin: Rep[Int], src_end: Rep[Int], dst_begin: Rep[Int], dst: Tensor[A]): Unit = {
      val mA = Backend.Const(manifest[A])
      Wrap[Unit](Adapter.g.reflectEffect(
        "tensor-data-copy", mA, Unwrap(src.data), Unwrap(dst.data), Backend.Const(src.dims), Unwrap(src_begin), Unwrap(src_end), Unwrap(dst_begin)
      )(
        Unwrap(src.data)
      )(
        Unwrap(dst.data)
      ))
    }
    def copy[A: Manifest](src: Tensor[A], dst: Tensor[A]): Unit = {
      assert(src.dims.product == dst.dims.product, s"Number of elements must match! ${src.dims}, ${dst.dims}")
      copy(src, 0, src.dims.product, 0, dst)
    }

    // Copying M vector into an NxM matrix, in a row-wise fashion
    def broadcast_copy[A: Manifest](src: Tensor[A], dst: Tensor[A]): Unit = {
      assert(src.dims.length == 1)
      assert(dst.dims.length == 2)
      assert(src.dims.head == dst.dims(1))

      val M: Int = src.dims.head
      val N: Int = dst.dims.head
      for (i <- 0 until N: Rep[Range]) {
        val begin = i * M
        Tensor.copy[A](src, 0, M, begin, dst)
      }
    }
  }
  class Tensor[A: Manifest: Ordering] (val dims: Seq[Int], var data: Rep[Array[A]], val allocType: AllocationType) {
    def createMemDesc(): Rep[MemDesc[A]] = {
      val mA = Backend.Const(manifest[A])
      Wrap[MemDesc[A]](Adapter.g.reflectRead("mem-desc", mA, Unwrap(data), Backend.Const(dims))(Unwrap(data)))
    }
    lazy val memDesc: Rep[MemDesc[A]] = createMemDesc()

    lazy val strides = dims.scanRight(1)(_ * _).tail
    lazy val totalSize = dims.product
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

    def fill(fillVal: Rep[A]): Unit = {
      val mA = Backend.Const(manifest[A])
      Wrap[Unit](Adapter.g.reflectWrite("tensor-fill", mA, Unwrap(data), Unwrap(fillVal), Backend.Const(dims))(Unwrap(data)))
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

    def unsafe_update(idx: Rep[Int], newVal: Rep[A]): Unit = {
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(data), Unwrap(idx), Unwrap(newVal))
      Wrap[Unit](Adapter.g.reflectWrite("tensor-update", unwrapped_xs:_*)(Unwrap(data)))
    }
    protected def transform[T: Manifest: Ordering](lhs: Rep[Array[T]], rhs: Rep[Array[T]], out: Rep[Array[T]], begin: Int, end: Int, op: String): Unit = {
      val mA = Backend.Const(manifest[T])
      Adapter.g.reflectEffect(
        "tensor-binary-transform-range", mA, Unwrap(lhs), Unwrap(rhs), Unwrap(out), Backend.Const((begin, end)), Backend.Const(op)
      )(
        Unwrap(lhs), Unwrap(rhs)
      )(
        Unwrap(out)
      )
    }
    def mapInplace(f: Rep[A] => Rep[A]): Unit = {
      transformRange(0, totalSize, f)
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
    protected def tensor_binary(rhs: Tensor[A], op: String): Tensor[A] = {
      checkDims(rhs.dims)
      val res = copy()
      res.tensor_binary_inplace(rhs, op)
      res
    }
    protected def tensor_binary_inplace(rhs: Tensor[A], op: String): Unit = {
      transform[A](this.data, rhs.data, this.data, 0, this.totalSize, op)
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

    def matmul(rhs: Tensor[A], bias_option: Option[Tensor[A]] = None, allocType: AllocationType): Tensor[A] = {
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
      bias_option match {
        case Some(bias) =>
          assert(bias.dims.length <= 2, s"Unsupported bias dimension: ${bias.dims}")
          bias.dims match {
            case d if d.length == 1 =>
              assert(d.head == N, s"1D bias should have dim $N")
            case d if d.length == 2 =>
              assert(bias.dims == Seq(M, N), s"2D Bias should have same dims as result ${bias.dims}, ${Seq(M, N)}")
          }
        case None =>

      }

      // bias might be in shape MxN or N. In latter case a broadcast is required
      val (result, beta) = bias_option match {
        case Some(bias) =>
          if (bias.dims == Seq(M, N)) {
            (bias.copy(), 1.0f)
          } else {
            val res = Tensor[A](Seq(M, N), allocType)
            for (i <- 0 until M: Rep[Range]) {
              val begin = i * N
              Tensor.copy[A](bias, 0, N, begin, res)
            }
            (res, 1.0f)
          }
        case None => (Tensor[A](Seq(M, N), allocType), 0.0f)
      }
      Wrap[Unit](
        Adapter.g.reflectEffect(
          "matrix-multiply", mA, Unwrap(data), Unwrap(rhs.data), Unwrap(result.data), Backend.Const(Seq(M, K, N)), Backend.Const(beta)
        )(
          Unwrap(data), Unwrap(rhs.data)
        )(
          Unwrap(result.data))
      )
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
      val Seq(oc, ic, kh, kw) = rhs.dims
      assert(kh == kw, s"Only square kernels are supported right now $kh $kw")
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
    def sum(): Rep[A] = accumulateRange(0, totalSize)
    def square(): Tensor[A] = {
      val output = copy()
      output.mapInplace(a => infix_*(a, a))
      output
    }
    def sqrt(): Tensor[A] = {
      val output = copy()
      output.mapInplace(__sqrt[A])
      output
    }
    def transform(f: Rep[A] => Rep[A]): Unit = transformRange(0, totalSize, f)
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
      val mA = Backend.Const(manifest[A])
      val dst = Tensor[A](dims, AllocationType.Intermediate)
      val epsilon = 0.00001f
      val avg = Tensor[A](Seq(c), AllocationType.Intermediate)
      val variance = Tensor[A](Seq(c), AllocationType.Intermediate)
      Wrap[Unit](
        Adapter.g.reflectEffect(
          "batchnorm-forward", mA+:Backend.Const(dims)+:Backend.Const(epsilon)+:Seq(this, avg, variance, gamma_beta, dst).map(a => Unwrap(a.memDesc)): _*
        )(
          Unwrap(data), Unwrap(gamma_beta.data)
        )(
          Seq(dst, avg, variance).map(a => Unwrap(a.data)): _*
        )
      )
//      dst.all_average(MPI.MPI_COMM_WORLD)
//      avg.all_average(MPI.MPI_COMM_WORLD)
//      variance.all_average(MPI.MPI_COMM_WORLD)
      (dst, avg, variance)
    }
    def sumT(): Tensor[A] = {
      val res = Tensor[A](Seq(1), AllocationType.Intermediate)
      val sumVal = sum()
      res.unsafe_update(0, sumVal)
      res
    }
    def sumRows(out: Tensor[A]): Tensor[A] = {
      assert(dims.length >= 2, "Tensor should be at least 2D. Perhaps you mean .sum() instead?")
      assert(dims.tail == out.dims, s"Out vector should have same trailing dimension as input ${dims.tail}, ${out.dims}")
      Adapter.g.reflectEffect(
        "tensor-sum-rows", Unwrap(data), Unwrap(out.data), Backend.Const(dims)
      )(
        Unwrap(data)
      )(
        Unwrap(out.data)
      )
      out
    }
    def avgT(): Tensor[A] = {
      val res = Tensor[A](Seq(1), AllocationType.Intermediate)
      val sumVal = sum()
      res.unsafe_update(0, infix_/(sumVal, dims.product.asInstanceOf[A]))
      res
    }
    def max(begin: Rep[Int] = 0, end: Rep[Int] = dims.product): Rep[A] = {
      Wrap[A](Adapter.g.reflectRead("tensor-max", Unwrap(data), Unwrap(begin), Unwrap(end))(Unwrap(data)))
    }
    def logsoftmax(target: Option[Tensor[A]] = None): Tensor[A] = {
      assert(dims.length == 2, "softmax currently only supports 2D tensors")
      assert(manifest[A].toString() == "Float", "softmax currently only supports floating point values")
      val (rows, rowSize) = dims.length match {
        case 1 => (1, dims.head)
        case _ => (dims.head, dims.product/dims.head)
      }
      val probs = target match {
        case Some(value) =>
          assert(value.dims.product == dims.product && value.dims.head == dims.head)
          value
        case None => copy()
      }
      Wrap[Unit](Adapter.g.reflectEffect(
        "logsoftmax-forward", Unwrap(memDesc), Unwrap(probs.memDesc), Backend.Const((rows, rowSize))
      )(
        Unwrap(data)
      )(
        Unwrap(probs.data)
      ))
      probs
    }
    def softmaxLoss(labels: Tensor[Int]): Rep[A] = {
      val probs = logsoftmax()
      __softmaxLoss(probs, labels)
    }
    def flatten(): Tensor[A] = {
      reshape(Seq(dims.product))
    }

    def reshape(newDims: Seq[Int]): Tensor[A] = {
      assert(dims.product == newDims.product, s"Number of elements must be preserved $dims, ${newDims}")
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(data), Backend.Const(dims), Backend.Const(AllocationType.Intermediate))
      new Tensor(
        newDims,
        Wrap[Array[A]](Adapter.g.reflectRead("tensor-copy", unwrapped_xs:_*)(Unwrap(data), STORE)),
        allocType
      )
    }

    def foreach(f: Rep[A] => Unit, begin: Rep[Int] = 0, end: Rep[Int] = dims.product): Unit = {
      val mA = Backend.Const(manifest[A])
      val block = Adapter.g.reify(exp => Unwrap(f(Wrap[A](exp))))
      Wrap[Unit](Adapter.g.reflectEffect(
        "tensor-foreach", mA, Unwrap(data), block, Unwrap(begin), Unwrap(end)
      )(
        (block.eff.rkeys + Unwrap(data)).toSeq: _*
      )(
        block.eff.wkeys.toSeq: _*
      ))
    }

    private def boolean_op(rhs: Tensor[A], boolean_operator: String): Tensor[Boolean] = {
      val res = Tensor[Boolean](dims, AllocationType.Intermediate)
      res.mapInplaceWithFlatIdx(idx =>
        Wrap[Boolean](Adapter.g.reflect(boolean_operator, Unwrap(unsafe_apply(idx)), Unwrap(rhs.unsafe_apply(idx))))
      )
      res
    }
    // Element-wise comparison on tensor
    def ==(rhs: Tensor[A]): Tensor[Boolean] = boolean_op(rhs, "==")
    def <=(rhs: Tensor[A]): Tensor[Boolean] = boolean_op(rhs, "<=")
    def >=(rhs: Tensor[A]): Tensor[Boolean] = boolean_op(rhs, ">=")
    def !=(rhs: Tensor[A]): Tensor[Boolean] = boolean_op(rhs, "!=")
    def <(rhs: Tensor[A]): Tensor[Boolean] = boolean_op(rhs, "<")
    def >(rhs: Tensor[A]): Tensor[Boolean] = boolean_op(rhs, ">")

    def fread(path: String, dtype: String): Unit = {
      val mA = Backend.Const(manifest[A])
      val offset = /*MPI.comm_rank(MPI.MPI_COMM_WORLD) * */ dims.product
      Wrap[Unit](Adapter.g.reflectEffect("tensor-fread", mA, Unwrap(data), Backend.Const(path), Backend.Const(dims), Backend.Const(dtype), Unwrap(offset))(Adapter.CTRL)(Unwrap(data)))
    }
    def all_average(comm: Rep[MPI_Comm]): Unit = {
      //MPI.All_average[A](comm, data, dims.product)
    }
  }
  def println(x: Tensor[_]): Unit = {
    println(x.data)
  }
}

object AllocationType extends Enumeration {
  type AllocationType = Value
  val Data, Gradient, Intermediate, Parameter = Value
}

// A trait that maps to dnnl::memory::dims. It holds the dimension of tensors at runtime
trait MemDims {
}
// A trait that maps to dnnl::memory. It holds the data pointer, format, and memory dims at runtime.
trait MemDesc[A] {
}


abstract class TensorDriverC[A: Manifest, B: Manifest] extends DslDriverC[A, B] with CPUTensorOps { q =>
  override val codegen = new CPUTensorCodeGen {
    override val IR: q.type = q
  }
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
        val mat3 = mat1.matmul(mat2, None, AllocationType.Data)
        println(mat3(0, 0))
        println(mat3(0, 1))
        println(mat3(0, 2))
      }
    }


    dslDriver.eval("5").foreach(println)
  }
}

