package tensor.ir

import lms.core.Backend.{Const, _}
import lms.core._
import lms.core.stub.Adapter.typeMap
import lms.core.stub._
import lms.macros.SourceContext
import tensor.ir.StagedMemoryAllocator.{Allocation, Deallocation, MemoryBlock, MemoryEvent}

import scala.collection.mutable


trait TensorOps extends Base with Equal with OrderingOps with PrimitiveOps with RandomOps {

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

  object Tensor {
    def apply[A: Manifest: Numeric](xs: Seq[Int])(implicit pos: SourceContext): Tensor[A] = {
      new Tensor(xs)
    }
    def fill[A: Manifest: Numeric](dims: Seq[Int], fillVal: A)(implicit pos: SourceContext): Tensor[A] = {
      val tensor = Tensor[A](dims)
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(tensor.data), Unwrap(fillVal), Backend.Const(dims))
      Wrap[Unit](Adapter.g.reflectWrite("tensor-fill", unwrapped_xs:_*)(Unwrap(tensor.data)))
      tensor
    }
    def zero[A: Manifest: Numeric](dims: Seq[Int])(implicit pos: SourceContext): Tensor[A] = {
      Tensor.fill[A](dims, 0.asInstanceOf[A])
    }
    def rand(dims: Seq[Int])(implicit pos: SourceContext): Tensor[Float] = {
      val tensor = Tensor[Float](dims)
      tensor.mapInplace(_ => randFloat())
      tensor
    }
  }
  class Tensor[A: Manifest: Numeric] (val dims: Seq[Int], var data: Rep[Array[A]]) {
    def infix_+(a: Rep[A], b: Rep[A]): Rep[A] = Wrap[A](Adapter.g.reflect("+", Unwrap(a), Unwrap(b)))
    def infix_-(a: Rep[A], b: Rep[A]): Rep[A] = Wrap[A](Adapter.g.reflect("-", Unwrap(a), Unwrap(b)))
    def infix_*(a: Rep[A], b: Rep[A]): Rep[A] = Wrap[A](Adapter.g.reflect("*", Unwrap(a), Unwrap(b)))
    def infix_/(a: Rep[A], b: Rep[A]): Rep[A] = Wrap[A](Adapter.g.reflect("/", Unwrap(a), Unwrap(b)))
    def sqrt(a: Rep[A]): Rep[A] = Wrap[A](Adapter.g.reflect("sqrt", Unwrap(a)))

    lazy val strides = dims.scanRight(1)(_ * _).tail
    def this(dims: Seq[Int]) {
      this(dims, null)
      data = {
        val mA = Backend.Const(manifest[A])
        Wrap[Array[A]](Adapter.g.reflectMutable("tensor-new", mA, Backend.Const(dims)))
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
    def copy(): Tensor[A] = {
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(data), Backend.Const(dims))
      new Tensor(
        dims,
        Wrap[Array[A]](Adapter.g.reflectRead("tensor-copy", unwrapped_xs:_*)(Unwrap(data), STORE))
      )
    }

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

    def matmul(rhs: Tensor[A]): Tensor[A] = {
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
      val result = Tensor[A](Seq(M, N))
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
      Seq[Int](dims.head, outChannels) ++ getConvOutputSize(Seq(kernelSize, kernelSize), padding, stride)
    }
    def conv(rhs: Tensor[A], pading: Int, stride: Int): Tensor[A] = {
      if (dims.length < 2) {
        throw new IllegalAccessError("Convolution can only be done on 3d or 4d tensors")
      }
      if (rhs.dims.length > dims.length) {
        throw new IllegalArgumentException(s"Kernel cannot be bigger than input, kernel dims: ${rhs.dims} input dims: $dims")
      }
      val outputDims = getConvOutputSize(rhs.dims, pading, stride)
      val output = Tensor[A](outputDims)
      val mA = Backend.Const(manifest[A])
      Wrap[Unit](Adapter.g.reflectEffect(
        "tensor-convolution",
        mA, Unwrap(data), Unwrap(rhs.data), Unwrap(output.data), Backend.Const(dims), Backend.Const(rhs.dims)
      )(Unwrap(data), Unwrap(rhs.data))(Unwrap(output.data)))
      output
    }
    def padd(padding: Int, paddingDims: Seq[Int]): Tensor[A] = {
      val newDims = new mutable.ArrayBuffer[Int](dims.length)
      dims.foreach(newDims += _)
      paddingDims.foreach(newDims(_) += 2*padding)
      val res = Tensor[A](newDims)
      Wrap[Unit](Adapter.g.reflectEffect(
        "tensor-padd", Unwrap(data), Unwrap(res.data), Backend.Const(dims), Backend.Const(padding), Backend.Const(paddingDims)
      )(
        Unwrap(data))(
        Unwrap(res.data)
      ))
      res
    }
    def conv2d(rhs: Seq[Tensor[A]], padding: Int, stride: Int): Tensor[A] = {
      assert(dims.length == 4, s"Convolution can only be done on 4d tensors $dims")
      assert(rhs.forall(_.dims.length == 3), s"Kernels must have dimmension of 3 ${rhs.map(_.dims)}")
      assert(rhs.tail.forall(_.dims == rhs.head.dims), s"All kernels must have the same size ${rhs.map(_.dims)}")

      val mA = Backend.Const(manifest[A])
      val data = padd(padding, Seq(2, 3)).data
      val outputSize = getConv2dOutputSize(dims(1), rhs.head.dims.head, rhs.head.dims(1), padding, stride)
      val output = Tensor[A](outputSize)
      Wrap[Unit](Adapter.g.reflectEffect(
        "tensor-convolution2d",
        Seq(mA, Unwrap(data), Unwrap(output.data), Backend.Const(dims), Backend.Const(outputSize), Backend.Const(rhs.head.dims)) ++ rhs.map(Unwrap(_)):_*
      )(Seq(Unwrap(data)) ++ rhs.map(Unwrap(_)):_*)(Unwrap(output.data)))
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
    def batchNormAvg(): Tensor[A] = {
      assert(this.dims.length == 4, "tensor for batch normal averaging should have 4 dimensions")
      val base = dims.product/dims(1)
      val res = Tensor.zero[A](Seq(dims(1), 1, 1))

      for (batch <- DataLoop(dims(0))) {
        val offsetBatch = batch * strides(0)
        for (channel <- DataLoop(dims(1))) {
          val offset = offsetBatch + channel * strides(1)
          res.data(channel) = infix_+(res.data(channel), accumulateRange(offset, offset + strides(1)))
        }
      }
      res.mapInplace(a => infix_/(a, base.asInstanceOf[A]))
      res
    }
    private def channelBroadcast(avg: Tensor[A], f: (Rep[A], Rep[A]) => Rep[A]): Tensor[A] = {
      assert(avg.dims.size == 1)
      assert(avg.dims.head == dims(1))
      val output = copy()

      for (batch <- DataLoop(dims(0))) {
        val offsetBatch = batch * strides(0)
        for (channel <- DataLoop(dims(1))) {
          val offset = offsetBatch + channel * strides(1)
          transformRange(offset, offset + strides(1), a => f(a, avg.unsafe_apply(channel)))
        }
      }
      output
    }
    private def subBatchAvg(avg: Tensor[A]): Tensor[A] = {
      channelBroadcast(avg, infix_-)
    }
    def batchNorm(gamma: Tensor[A], beta: Tensor[A]): (Tensor[A], Tensor[A], Tensor[A], Tensor[A]) = {
      assert(gamma.dims == Seq(dims(1)), s"Gamma should be same size as channels $dims")
      assert(beta.dims == gamma.dims, s"Beta and Gamma should have same dims ${beta.dims} ${gamma.dims}")
      val saveMean = batchNormAvg()
      val diff = subBatchAvg(saveMean)
      val saveInvVariance = diff.square().batchNormAvg()
      val epsilon = 0.00001f
      val xhat = diff.channelBroadcast((saveInvVariance + epsilon.asInstanceOf[A]).sqrt(), infix_/)
      val outy = xhat.channelBroadcast(gamma, infix_*).channelBroadcast(beta, infix_+)
      // runningMean and runningVariance should also be updated???
      (outy, xhat, saveMean, saveInvVariance)
    }
    def flatten(): Tensor[A] = {
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(data), Backend.Const(dims))
      new Tensor(
        Seq(dims.product),
        Wrap[Array[A]](Adapter.g.reflectRead("tensor-copy", unwrapped_xs:_*)(Unwrap(data), STORE))
      )
    }
  }
  def println(x: Tensor[_]): Unit = {
    println(x.data)
  }
}

trait BaseGenTensorOps extends DslGenC with RandomOpsCodegen {
  override def init(g: Graph): Graph = {
    val graph = memoryPlanning(g)
    super.init(graph)
  }
  def memoryPlanning(g: Graph): Graph = {
    val traverser = new MemoryPlanningTraverser()
    traverser(g)
    val events = traverser.events.values
    val allocationPlan = StagedMemoryAllocator.allocate(events.toSeq)

    val transformer = new MemoryPlanningTransformer(allocationPlan, traverser.reusedSyms.toMap)
    val newGraph = transformer.transform(g)
    typeMap = transformer.newTypeMap
    newGraph
  }
  doRename = false
//  val _shouldInline = shouldInline
  var totalMemory: Int = 0
  var allocationPlan: Map[Int, MemoryBlock] = Map()
  registerHeader("<string.h>", "<algorithm>")
  registerHeader("<cblas.h>")
  registerHeader("<sys/mman.h>", "<unistd.h>")
  registerLibrary("-L/opt/OpenBLAS/lib", "-I/opt/OpenBLAS/include", "-lopenblas", "-g")
  registerDatastructures("heap"){
    emit("char *heap = NULL;")
  }
  registerInit("heap_init") {
    emit("heap = (char*)get_mem(1024*1024*1024);")
  }
  registerTopLevelFunction("tensor_copy"){
    emit(
      """
        |static void *memdup(void* source, size_t bytes) {
        |   void *copy = malloc(bytes);
        |   memcpy(copy, source, bytes);
        |   return copy;
        |}
        |""".stripMargin)
  }
  registerTopLevelFunction("bump_allocate") {
    emit(
      """
        |static void *bump_allocate(size_t size) {
        |   void *old_heap = heap;
        |   heap += size;
        |   return old_heap;
        |}
        |""".stripMargin
    )
  }
  registerTopLevelFunction("get_mem") {
    emit(
      """
        |void *get_mem(size_t size) {
        |  size_t page_size = getpagesize();
        |  size = (size + page_size - 1) / page_size * page_size;
        |  void *p = mmap(NULL, size, PROT_READ | PROT_WRITE,
        |                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        |  return p;
        |}
        |""".stripMargin)
  }

  val is_tensor_new_ops = Set("tensor-new", "heap-offset")

  override def shallow(node: Node): Unit = node match {
    case Node(s, "tensor-new", Const(manifest: Manifest[_])::Backend.Const(dims: Seq[Int])::Nil, eff) =>
      emit(s"malloc(${dims.product} * sizeof(${remap(manifest)}))")
    case Node(s, "heap-offset", Const(manifest: Manifest[_])::Const(blk: MemoryBlock)::src, eff) =>
      if (src.isEmpty)
        emit(s"((${remap(manifest)}*)(heap+${blk.begin} * sizeof(${remap(manifest)})))")
      else
        shallow(src.head)

    case Node(s, "tensor-apply", List(_, tensor, Const(idx: Seq[Int]), Const(dims: Seq[Int])), _) =>
      val sizes = dims.scanRight(1)(_ * _).tail
      shallow(tensor)
      emit(s"[${idx.zip(sizes).map{case (a, b) => a*b}.sum}]")
    case Node(s, "tensor-apply", List(_, tensor, idx), _) =>
      // Comming from unsafe_apply
      shallow(tensor)
      emit("[")
      shallow(idx)
      emit("]")
    case Node(s, "tensor-update", List(_, tensor, Const(idx: Seq[Int]), newVal, Const(dims: Seq[Int])), _) =>
      val sizes = dims.scanRight(1)(_ * _).tail
      shallow(tensor)
      emit(s"[${idx.zip(sizes).map{case (a, b) => a*b}.sum}] = ")
      shallow(newVal)
    case Node(s, "tensor-update", List(_, tensor, idx, newVal), _) =>
      shallow(tensor)
      emit("[")
      shallow(idx)
      emit("] = ")
      shallow(newVal)
    case Node(s, "tensor-fill", List(mA, tensor, fillVal, Const(dims: Seq[Int])), _) =>
      val totalSize = dims.product
      emit("std::fill(")
      shallow(tensor)
      emit(", ")
      shallow(tensor)
      emit(s" + $totalSize, ${quote(fillVal)})")


    case Node(s, "tensor-copy", List(mA, tensor, Const(dims: Seq[Int])), _) =>
      val manifest = mA match {case Const(mani: Manifest[_]) => mani}
      val totalSize = dims.product
      val byteSize = s"$totalSize * sizeof(${remap(manifest)})"
      emit(s"((${remap(manifest)}*)")
      emit("(memdup(")
      shallow(tensor)
      emit(", ")
      emit(byteSize.toString)
      emit(")))")
    case Node(s, "heap-offset-copy", Const(manifest: Manifest[_])::tensor::Const(blk: MemoryBlock)::Const(dims: Seq[Int])::_, eff) =>
      emit(s"((${remap(manifest)} *)memcpy(heap+${blk.begin} * sizeof(${remap(manifest)}), ")
      shallow(tensor)
      val byteSize = s"${dims.product} * sizeof(${remap(manifest)})"
      emit(s", $byteSize))")

    case Node(s, "matrix-multiply", List(mA, lhs, rhs, result, Const(Seq(m: Int, k: Int, n: Int))), _) =>
      if (mA.toString != "Float") {
        throw new RuntimeException(s"Only floating point values are supported: ${mA.toString}")
      }
      emit(s"cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, $m, $n, $k, 1, ")
      shallow(lhs)
      emit(s", $m, ")
      shallow(rhs)
      emit(s", $k, 0, ")
      shallow(result)
      emit(s", $m)")

    case n @ Node(s,"P",List(x),_) =>
      emit("""printf("""")
      emit(format(x))
      emit("""\n", """) // Should look like <BEGIN>\n", <END>
      shallow(x)
      emit(")")
    case Node(s, "tensor-accumulate-range", List(Const(mA: Manifest[_]), data, begin, end), _) =>
      emit("std::accumulate(")
      shallow(data)
      emit("+")
      shallow(begin)
      emit(", ")
      shallow(data)
      emit("+")
      shallow(end)
      emit(s", ${remap(mA)}(0))")
    case Node(s, "tensor-transform-range", List(Const(mA: Manifest[_]), data, block: Block, begin, end), _) =>
      assert(block.in.length == 1)
      emit("std::transform(")
      shallow(data)
      emit("+")
      shallow(begin)
      emit(", ")
      shallow(data)
      emit("+")
      shallow(end)
      emit(", ")
      shallow(data)
      emit(s", [&](${remap(mA)} ")
      shallow(block.in.head)
      emit(")")
      quoteBlockPReturn(traverse(block))
      emit(")")

    case Node(s, "tensor-transform-index", List(Const(mA: Manifest[_]), data, block: Block, Const(dims: Seq[Int])), _) =>
      assert(block.in.length == 1)
      val totalSize = dims.product
      val counter = s"i${s.n}"
      emit(s"for(size_t $counter = 0; $counter < $totalSize; ++$counter) {")
      emit(s"const size_t ")
      shallow(block.in.head)
      emit(s" = $counter;")
      shallow(data)
      emit("[")
      shallow(block.in.head)
      emit("] = ")
      quoteBlockP(traverse(block))
      emit(";}")
    case Node(s, "tensor-convolution", List(mA, data, kernel, output, Const(dims: Seq[Int]), Const(kernelDims: Seq[Int])), _) =>
      // TODO implement convolution, this is just a stub
      emit("/*Stub for tensor convolution TODO implement this*/")
    case Node(s, "max", List(lhs, rhs), _) =>
      emit(s"std::max<${remap(typeMap(s))}>(")
      shallow(lhs)
      emit(", ")
      shallow(rhs)
      emit(")")
    case _ => super.shallow(node)
  }
  def format(x: Def): String = x match {
    case exp: Exp => exp match {
      case s@Sym(_) => typeMap(s).toString match {
        case m if m.matches("""Array\[.*\]""") =>
          "%p"
        case "Float" =>
          "%f"
        case "Int" =>
          "%d"
      }
      case Const(_: Int) => "%d"
      case Const(_: Float) => "%f"
      case _ => "%s"
    }
    case _ => "%s"
  }
}

abstract class TensorDriverC[A: Manifest, B: Manifest] extends DslDriverC[A, B] with TensorOps { q =>
  override val codegen = new BaseGenTensorOps {
    override val IR: q.type = q
  }
  lazy val g: Graph = Adapter.program(Adapter.g.reify(x => Unwrap(wrapper(Wrap[A](x)))))
}

class MemoryPlanningTransformer(val allocationPlan: Map[Int, MemoryBlock], val reusedSyms: Map[Sym, Sym]) extends Transformer {
  g = Adapter.mkGraphBuilder()
  val totalMemory: Int = {
    val maxBlock = allocationPlan.values.maxBy(_.begin)
    maxBlock.begin + maxBlock.size
  }

  lazy val newTypeMap: mutable.Map[Exp, Manifest[_]] = {
    val newMap: mutable.Map[Exp, Manifest[_]] = new mutable.HashMap[Exp, Manifest[_]]
    typeMap.foreach{
      case (exp, mani) =>
        symMap.get(exp) match {
          case Some(value) => newMap(value) = mani
          case None => newMap.getOrElseUpdate(exp, mani)
        }
    }
    newMap
  }

  val symMap: mutable.Map[Exp, Exp] = new mutable.HashMap[Exp, Exp]()
  @scala.annotation.tailrec
  private final def getSrc(s: Sym): Sym = reusedSyms.get(s) match {
    case Some(value) => getSrc(value)
    case None => s
  }
  override def transform(n: Node): Exp = n match {
    case Node(s, "tensor-new", List(mA, dims), eff) =>
      val exp = g.reflectEffect("heap-offset", mA, Const(allocationPlan(s.n)))(
        eff.rkeys.map(transform).toSeq: _*
      )(
        eff.wkeys.map(transform).toSeq: _*
      )
      symMap(s) = exp
      exp
    case Node(s, "tensor-copy", List(mA, tensor, dims), eff) =>
      val exp = if (!allocationPlan.contains(s.n)) {
        assert(reusedSyms.contains(s))
        val src = getSrc(s)
        val arg = tensor match {
          case b @ Block(_,_,_,_) =>
            transform(b)
          case s : Exp =>
            transform(s)
          case a =>
            a
        }
        val exp = g.reflectEffect("heap-offset", mA, Const(allocationPlan(src.n)), arg)(
          eff.rkeys.map(transform).toSeq: _*
        )(
          eff.wkeys.map(transform).toSeq: _*
        )
        exp
      } else {
        val src = symMap(tensor.asInstanceOf[Sym])
        val exp = g.reflectEffect("heap-offset-copy", mA, src, Const(allocationPlan(s.n)), dims)(
          eff.rkeys.map(transform).toSeq: _*
        )(
          eff.wkeys.map(transform).toSeq: _*
        )
        exp
      }

      symMap(s) = exp
      exp
    case Node(s, _, _, _) =>
      val newNode = super.transform(n)
      symMap(s) = newNode
      newNode
  }
}

class MemoryPlanningTraverser extends Traverser {
  var time: Int = 0
  def getTime(): Int = {
    time+=1
    time
  }
  class MemoryRequest(val allocatedTime: Int, var deallocatedTime: Int, var lastUseSym: Sym, val size: Int, val src: Sym, val isCopy: Boolean = false) {
    override def toString: String = s"[$allocatedTime, $deallocatedTime]: $size"
  }

  override def apply(g: Graph): Unit = {
    @scala.annotation.tailrec
    def getSrc(s: Sym): Sym = reusedSyms.get(s) match {
      case Some(value) => getSrc(value)
      case None => s
    }
    super.apply(g)
    requests.foreach{case (n: Sym, req) =>
      val srcSym = getSrc(req.src)
      if (srcSym != null){
        val srcReq = requests(srcSym)
        if (req.isCopy && srcReq.lastUseSym == n) {
          srcReq.deallocatedTime = req.deallocatedTime
          srcReq.lastUseSym = req.lastUseSym
          requests.remove(n)
          reusedSyms(n) = srcSym
          println(s"$n is reusing ${req.src}'s memory")
        }
      }
    }
  }
  val reusedSyms = mutable.HashMap[Sym, Sym]()
  val requests = scala.collection.mutable.HashMap[Sym, MemoryRequest]()

  lazy val events = {
    val bst = scala.collection.mutable.TreeMap[Int, MemoryEvent]()
    requests.foreach{
      case (key, req) =>
        bst(req.allocatedTime) = Allocation(key.n, req.size)
        bst(req.deallocatedTime) = Deallocation(key.n, req.size, req.lastUseSym)
    }
    bst
  }
  override def traverse(n: Node): Unit = {
    n match {
      case Node(n, "tensor-new", List(mA, Backend.Const(dims: Seq[Int])), _) =>
        requests(n) = new MemoryRequest(getTime(), getTime(), n, dims.product, null)
      case Node(n, "tensor-copy", _:: src :: dims :: Nil, eff) =>
        val size = (dims match {case Backend.Const(seq: Seq[Int]) => seq}).product
        src match {
          case exp: Exp => exp match {
            case s@Sym(_) =>
              requests(n) = new MemoryRequest(getTime(), getTime(), n, size, s, true)
              requests(s).deallocatedTime = getTime()
              requests(s).lastUseSym = n
          }
        }
      case Node(n, _, _, eff) =>
        eff.rkeys.foreach{
          case s@Sym(_) =>
          requests.get(s) match {
            case Some(request) =>
              request.deallocatedTime = getTime()
              request.lastUseSym = n
            case None =>
          }
        case _ =>
        }
    }
  }
}

object Runer {

  def main(args: Array[String]) {
    val dslDriver = new TensorDriverC[String,Unit] {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        var tensor = Tensor.fill[Float](Seq(1, 2, 3), 4.0)
        tensor = tensor + 1
        tensor = tensor - 1
        tensor = tensor + 1
        tensor = tensor - 1
        val tensor2 = Tensor[Float](Seq(1, 2, 3))
        tensor2.mapInplaceWithFlatIdx(idx => idx)
        println(tensor)

        println(tensor(0, 1, 2))
        println(tensor.copy()(0, 1, 2))
        println((tensor add tensor)(0, 0, 0))
        println((tensor+tensor(0, 0, 0))(0, 1, 2))
        println(tensor2(0, 1, 2))

        val mat1 = Tensor.fill[Float]((Seq(3, 3)), 0)
        mat1(Seq(0, 0)) = 1.0
        mat1(Seq(1, 1)) = 1.0
        mat1(Seq(2, 2)) = 1.0
        val mat2 = Tensor[Float](Seq(3, 3))
        mat2.mapInplaceWithFlatIdx(idx => idx)
        val mat3 = mat1.matmul(mat2)
        println(mat3(0, 0))
        println(mat3(0, 1))
        println(mat3(0, 2))
      }
    }


    dslDriver.eval("5").foreach(println)
  }
}

