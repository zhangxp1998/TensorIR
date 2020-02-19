package tensor.ir

import lms.core.Backend.{Const, _}
import lms.core._
import lms.core.stub.Adapter.typeMap
import lms.core.stub._
import lms.macros.SourceContext
import tensor.ir.StagedMemoryAllocator.{Allocation, Deallocation, MemoryBlock, MemoryEvent}

import scala.collection.mutable


trait TensorOps extends Base with Equal {
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
  }
  class Tensor[A: Manifest: Numeric] (val dims: Seq[Int], var data: Rep[Array[A]]) {
    def this(dims: Seq[Int]) {
      this(dims, null)
      data = {
        val mA = Backend.Const(manifest[A])
        val unwrapped_xs: Seq[Backend.Def] = Seq(mA) ++ dims.map(i => Backend.Const(i))
        Wrap[Array[A]](Adapter.g.reflectMutable("tensor-new", unwrapped_xs:_*))
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
      val mA = Backend.Const(manifest[A])
      val block = Adapter.g.reify(exp => Unwrap(f(Wrap[A](exp))))
      Wrap[Unit](Adapter.g.reflectEffect(
        "tensor-transform", mA, Unwrap(data), block, Backend.Const(dims)
      )(
        (block.eff.rkeys + Unwrap(data)).toSeq: _*
      )(
        (block.eff.wkeys + Unwrap(data)).toSeq: _*
      ))
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
  }
  def println(x: Tensor[_]): Unit = {
    println(x.data)
  }
}

trait BaseGenTensorOps extends DslGenC {
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

  val is_tensor_new_ops = Set("tensor-new", "heap-offset")

  override def shallow(node: Node): Unit = node match {
    case Node(s, "tensor-new", Const(manifest: Manifest[_])::tail, eff) =>
      val dims = tail.map{case Const(i: Int) => i}
      emit(s"malloc(${dims.product} * sizeof(${remap(manifest)}))")
    case Node(s, "heap-offset", Const(manifest: Manifest[_])::Const(blk: MemoryBlock)::_, eff) =>
      emit(s"((${remap(manifest)}*)(heap+${blk.begin} * sizeof(${remap(manifest)})))")

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

    case n @ Node(s,"P",List(x),_) =>
      emit("""printf("""")
      emit(format(x))
      emit("""\n", """) // Should look like <BEGIN>\n", <END>
      shallow(x)
      emit(")")
    case Node(s, "tensor-transform", List(Const(mA: Manifest[_]), data, block: Block, Const(dims: Seq[Int])), _) =>
      assert(block.in.length == 1)
      val totalSize = dims.product
      emit("std::transform(")
      shallow(data)
      emit(", ")
      shallow(data)
      emit(s" + $totalSize, ")
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
    override def init(g: Graph): Graph = {
      val graph = memoryPlanning(g)
      super.init(graph)
    }
    def memoryPlanning(g: Graph): Graph = {
      val traverser = new MemoryPlanningTraverser()
      traverser(g)
      val events = traverser.events.values
      val allocationPlan = StagedMemoryAllocator.allocate(events.toSeq)

      val transformer = new MemoryPlanningTransformer(allocationPlan)
      val newGraph = transformer.transform(g)
      typeMap = transformer.newTypeMap
      newGraph
    }
  }
  lazy val g: Graph = Adapter.program(Adapter.g.reify(x => Unwrap(wrapper(Wrap[A](x)))))
}

class MemoryPlanningTransformer(val allocationPlan: Map[Int, MemoryBlock]) extends Transformer {
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
  override def transform(n: Node): Exp = n match {
    case Node(s, "tensor-new", Const(mA: Manifest[_])::dims, eff) =>
      val exp = g.reflectEffect("heap-offset", Const(mA)::Const(allocationPlan(s.n))::dims:_*)(
        eff.rkeys.map(transform).toSeq: _*
      )(
        eff.wkeys.map(transform).toSeq: _*
      )
      symMap(s) = exp
      exp
    case Node(s, "tensor-copy", List(mA, tensor, dims), eff) =>
      val src = symMap(tensor.asInstanceOf[Sym])
      val exp = g.reflectEffect("heap-offset-copy", mA, src, Const(allocationPlan(s.n)), dims)(
        eff.rkeys.map(transform).toSeq: _*
      )(
        eff.wkeys.map(transform).toSeq: _*
      )
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
  class MemoryRequest(val allocatedTime: Int, var deallocatedTime: Int, var lastUseSym: Sym, val size: Int) {
    override def toString: String = s"[$allocatedTime, $deallocatedTime]: $size"
  }

  val requests = scala.collection.mutable.HashMap[Int, MemoryRequest]()

  lazy val events = {
    val bst = scala.collection.mutable.TreeMap[Int, MemoryEvent]()
    requests.foreach{
      case (key, req) =>
        bst(req.allocatedTime) = Allocation(key, req.size)
        bst(req.deallocatedTime) = Deallocation(key, req.size, req.lastUseSym)
    }
    bst
  }
  override def traverse(n: Node): Unit = {
    n match {
      case Node(n, "tensor-new", Const(manifest: Manifest[_]):: dims, _) =>
        val size = dims.map{case Backend.Const(i: Int) => i}.product
        requests(n.n) = new MemoryRequest(getTime(), getTime(), n, size)
      case Node(n, "tensor-copy", _:: src :: dims :: Nil, _) =>
        val size = (dims match {case Backend.Const(seq: Seq[Int]) => seq}).product
        requests(n.n) = new MemoryRequest(getTime(), getTime(), n, size)
        src match {
          case exp: Exp => exp match {
            case Sym(ref) =>
              requests(ref).deallocatedTime = getTime()
              requests(ref).lastUseSym = n
          }
        }
      case Node(n, _, _, eff) =>
        eff.rkeys.foreach{
          case Sym(ref) =>
          requests.get(ref) match {
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
        val tensor = Tensor[Float](Seq(1, 2, 3))
        tensor.mapInplace(_ => 4.0f)
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

