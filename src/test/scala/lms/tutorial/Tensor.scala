package scala.lms.tutorial


import lms.core._
import lms.util._
import lms.core.stub._
import lms.core.Backend._
import lms.core.stub.Adapter.typeMap
import lms.core.virtualize
import lms.core.utils.time
import lms.macros.{RefinedManifest, SourceContext}

trait Tensor[A] {

}

trait TensorOps { b: Base =>
  object Tensor {
    def apply[A: Manifest](xs: Seq[Int])(implicit pos: SourceContext): Rep[Tensor[A]] = {
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA) ++ xs.map(i => Backend.Const(i))
      Wrap[Tensor[A]](Adapter.g.reflectWrite("tensor-new", unwrapped_xs:_*)(STORE))
    }
    def fill[A: Manifest](dims: Seq[Int], fillVal: A)(implicit pos: SourceContext): Rep[Tensor[A]] = {
      val tensor = Tensor[A](dims)
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(tensor), Unwrap(fillVal), Backend.Const(dims))
      Wrap[Unit](Adapter.g.reflectWrite("tensor-fill", unwrapped_xs:_*)(Unwrap(tensor)))
      tensor
    }
  }

  implicit class TensorOps[A: Manifest](tensor: Rep[Tensor[A]]) {
    lazy val dims: Seq[Int] = Adapter.g.findDefinition(Unwrap(tensor)) match {
      case Some(Node(n, "tensor-new", _:: dims, _)) => dims.map{case Backend.Const(i: Int) => i}
      case Some(Node(n, "tensor-copy", _:: _ :: dims :: Nil, _)) => dims match {case Backend.Const(seq: Seq[Int]) => seq}
    }
    def checkIdx(idx: Seq[Int]) = {
      assert(dims.length == idx.length, s"Tensor index $idx does not match dimension $dims")
      assert(idx.zip(dims).forall{case (a, b) => a < b}, s"Tensor index $idx is out of bounds for dimension $dims")
    }
    def apply(idx: Int*): Rep[A] = {
      checkIdx(idx)
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(tensor), Backend.Const(idx), Backend.Const(dims))
      Wrap[A](Adapter.g.reflectRead("tensor-apply", unwrapped_xs:_*)(Unwrap(tensor)))
    }

    def update(idx: Seq[Int], newVal: A): Unit = {
      update(idx, Const(newVal))
    }

    def update(idx: Seq[Int], newVal: Rep[A]): Unit = {
      checkIdx(idx)
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(tensor), Backend.Const(idx), Unwrap(newVal), Backend.Const(dims))
      Wrap[Unit](Adapter.g.reflectWrite("tensor-update", unwrapped_xs:_*)(Unwrap(tensor)))
    }
    private def unsafe_apply(idx: Rep[Int]): Rep[A] = {
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(tensor), Unwrap(idx))
      Wrap[A](Adapter.g.reflectRead("tensor-apply", unwrapped_xs:_*)(Unwrap(tensor)))
    }

    private def unsafe_update(idx: Rep[Int], newVal: Rep[A]): Unit = {
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(tensor), Unwrap(idx), Unwrap(newVal))
      Wrap[Unit](Adapter.g.reflectWrite("tensor-update", unwrapped_xs:_*)(Unwrap(tensor)))
    }

    def mapInplace(f: Rep[A] => Rep[A]): Unit = {
      val totalSize = dims.product
      for(i <- 0 until totalSize: Rep[Range]) {
        unsafe_update(i, f(unsafe_apply(i)))
      }
    }
    def mapInplaceWithFlatIdx(f: (Rep[A], Rep[Int]) => Rep[A]): Unit = {
      val totalSize = dims.product
      for(i <- 0 until totalSize: Rep[Range]) {
        unsafe_update(i, f(unsafe_apply(i), i))
      }
    }

    def copy(): Rep[Tensor[A]] = {
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(tensor), Backend.Const(dims))
      Wrap[Tensor[A]](Adapter.g.reflectRead("tensor-copy", unwrapped_xs:_*)(Unwrap(tensor)))
    }

    private def binary_broadcast(op: String, rhs: Rep[A]): Rep[Tensor[A]] = {
      val mA = Backend.Const(manifest[A])
      val result = tensor.copy()
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(result), Backend.Const(op), Unwrap(rhs), Backend.Const(dims))
      Wrap[Unit](Adapter.g.reflectWrite("tensor-binary-broadcast", unwrapped_xs:_*)(Unwrap(result)))
      result
    }

    def +(rhs: Rep[A]): Rep[Tensor[A]] = {
      binary_broadcast("+=", rhs)
    }
    def -(rhs: Rep[A]): Rep[Tensor[A]] = {
      binary_broadcast("-=", rhs)
    }
    def *(rhs: Rep[A]): Rep[Tensor[A]] = {
      binary_broadcast("*=", rhs)
    }
    def /(rhs: Rep[A]): Rep[Tensor[A]] = {
      binary_broadcast("/=", rhs)
    }

    def matmul(rhs: Rep[Tensor[A]]): Rep[Tensor[A]] = {
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
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(tensor), Unwrap(rhs), Unwrap(result), Backend.Const(Seq(M, K, N)))
      Wrap[Unit](Adapter.g.reflectEffect("matrix-multiply", unwrapped_xs:_*)(Unwrap(tensor), Unwrap(rhs))(Unwrap(result)))
      result
    }
  }
}

trait BaseGenTensorOps extends DslGenC {
  doRename = false
//  val _shouldInline = shouldInline
  registerHeader("<string.h>")
  registerHeader("<cblas.h>")
  registerLibrary("-L/opt/OpenBLAS/lib", "-I/opt/OpenBLAS/include", "-lopenblas")
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

  override def shallow(node: Node): Unit = node match {
    case Node(s, "tensor-new", rhs, eff) =>
      val manifest = rhs.head match {case Const(mani: Manifest[_]) => mani}
      val dims = rhs.tail.map{case Const(i: Int) => i}
      emit(s"malloc(${dims.product} * sizeof(${remap(manifest)}))")
    case Node(s, "tensor-apply", List(_, tensor, Const(idx: Seq[Int]), Const(dims: Seq[Int])), _) =>
      val sizes = dims.scanRight(1)(_ * _).tail
      shallow(tensor)
      emit(s"[${idx.zip(sizes).map{case (a, b) => a*b}.sum}]")
    case Node(s, "tensor-apply", List(_, tensor, idx), _) =>
      // Comming from unsafe_apply
      shallow(tensor)
      emit(s"[${shallow(idx)}]")
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
      val loopCounter = "i" + s.n
      emitln(s"for (int $loopCounter = 0; $loopCounter < $totalSize; $loopCounter ++) {")
      shallow(tensor)
      emitln(s"[$loopCounter] = ")
      shallow(fillVal)
      emitln(";}")

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
    case Node(s, "tensor-binary-broadcast", List(mA, tensor, Backend.Const(op: String), rhs, Backend.Const(dims: Seq[Int])), _) =>
      val totalSize = dims.product
      val loopCounter = "i" + s.n
      emit(
        s"""
          |for (int $loopCounter = 0; $loopCounter < $totalSize; $loopCounter ++) {
          |
          |""".stripMargin)
      shallow(tensor)
      emit(s"[$loopCounter] $op ")
      shallow(rhs)
      emit(
        """
          |;
          |}
          |""".stripMargin)

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
      emit(s", $m);")

    case n @ Node(s,"P",List(x),_) =>
      emit("""printf("""")
      emit(format(x))
      emit("""\n", """) // Should look like <BEGIN>\n", <END>
      shallow(x)
      emit(")")

    case _ => super.shallow(node)
  }
  def format(x: Def): String = x match {
    case exp: Exp => exp match {
      case s@Sym(_) => typeMap(s).toString match {
        case m if m.contains("scala.lms.tutorial.Tensor[") =>
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

class MemoryPlanningTraverser extends Traverser {
  var time: Int = 0
  def getTime(): Int = {
    time+=1
    time
  }
  class MemoryRequest(val allocatedTime: Int, var deallocatedTime: Int, var lastUseSym: Sym, val size: Int) {
    override def toString: String = s"[$allocatedTime, $deallocatedTime]: $size"
  }

  class MemoryEvent {

  }
  case class Allocation(id: Int, size: Int) extends MemoryEvent
  case class Deallocation(id: Int, size: Int, afterSym: Sym) extends MemoryEvent

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
      case Node(n, "tensor-new", _:: dims, _) =>
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
        val tensor = Tensor.fill[Float](Seq(1, 2, 3), 4.0)
        val tensor2 = Tensor[Float](Seq(1, 2, 3))
        tensor2.mapInplaceWithFlatIdx((_, idx) => idx)
        println(tensor)

        println(tensor(0, 1, 2))
        println(tensor.copy()(0, 1, 2))
        println((tensor+tensor(0, 0, 0))(0, 1, 2))
        println(tensor2(0, 1, 2))

        val mat1 = Tensor[Float](Seq(3, 3))
        mat1(Seq(0, 0)) = 1.0
        mat1(Seq(1, 1)) = 1.0
        mat1(Seq(2, 2)) = 1.0
        val mat2 = Tensor[Float](Seq(3, 3))
        mat2.mapInplaceWithFlatIdx((_, idx) => idx)
        val mat3 = mat1.matmul(mat2)
        println(mat3(0, 0))
        println(mat3(0, 1))
        println(mat3(0, 2))
      }
    }

    val traverser = new MemoryPlanningTraverser()
    Adapter.typeMap = new scala.collection.mutable.HashMap[lms.core.Backend.Exp, Manifest[_]]()
    traverser(dslDriver.g)
    traverser.events.values.foreach(println)

    dslDriver.eval("5")
  }
}

