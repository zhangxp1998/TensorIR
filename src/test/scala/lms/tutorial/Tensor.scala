package scala.lms.tutorial


import lms.core._
import lms.util._
import lms.core.stub._
import lms.core.Backend._
import lms.core.virtualize
import lms.core.utils.time
import lms.macros.{RefinedManifest, SourceContext}
import scala.collection.mutable.ArrayBuffer
import scala.reflect.AnyValManifest

trait Tensor[A] {

}

trait TensorOps { b: Base =>
  object Tensor {
    def apply[A: Manifest](xs: Seq[Int])(implicit pos: SourceContext): Rep[Tensor[A]] = {
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA) ++ xs.map(i => Backend.Const(i))
      Wrap[Tensor[A]](Adapter.g.reflect("tensor-new", unwrapped_xs:_*))
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
//        Adapter.g.reify((e => Unwrap(f(Wrap[A](e)))): Backend.Exp => Backend.Exp)
      }
    }

    def copy(): Rep[Tensor[A]] = {
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(tensor), Backend.Const(dims))
      Wrap[Tensor[A]](Adapter.g.reflectRead("tensor-copy", unwrapped_xs:_*)(Unwrap(tensor)))
    }

    def +(rhs: Rep[A]): Rep[Tensor[A]] = {
      val mA = Backend.Const(manifest[A])
      val result = tensor.copy()
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(result), Unwrap(rhs), Backend.Const(dims))
      Wrap[Unit](Adapter.g.reflectWrite("tensor-add-broadcast", unwrapped_xs:_*)(Unwrap(result)))
      result
    }
  }
}

trait BaseGenTensorOps extends DslGenC {
  registerHeader("<string.h>")
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
      emit(quote(newVal))
    case Node(s, "tensor-fill", List(mA, tensor, fillVal, Const(dims: Seq[Int])), _) =>
      val totalSize = dims.product
      val loopCounter = "i" + s.n
      emitln(s"for (int $loopCounter = 0; $loopCounter < $totalSize; $loopCounter ++) {")
      shallow(tensor)
      emitln(s"[$loopCounter] = ${quote(fillVal)};")
      emitln("}")

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
    case Node(s, "tensor-add-broadcast", List(mA, tensor, rhs, Backend.Const(dims: Seq[Int])), _) =>
      val totalSize = dims.product
      val loopCounter = "i" + s.n
      emit(
        s"""
          |for (int $loopCounter = 0; $loopCounter < $totalSize; $loopCounter ++) {
          |
          |""".stripMargin)
      shallow(tensor)
      emit(s"[$loopCounter] += ")
      shallow(rhs)
      emit(
        """
          |;
          |}
          |""".stripMargin)

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

trait TensorDriverC[A, B] extends DslDriverC[A, B] with TensorOps { q =>
  override val codegen = new BaseGenTensorOps {
    override val IR: q.type = q
  }
}
object Runer {

  def main(args: Array[String]) {
    val dslDriver = new TensorDriverC[String,Unit] {
      override def snippet(x: Rep[String]): Rep[Unit] = {
        val tensor = Tensor.fill[Float](Seq(1, 2, 3), 4.0)
        println(tensor)

        println(tensor(0, 1, 2))
        println(tensor.copy()(0, 1, 2))
        println((tensor+tensor(0, 0, 0))(0, 1, 2))
        println(123)
      }
    }
    dslDriver.eval("5")
  }
}

