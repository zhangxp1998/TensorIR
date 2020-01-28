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
      val tensor: Rep[Tensor[A]] = Tensor[A](dims)
      val xs: Seq[Backend.Def] = Seq(Unwrap(tensor), Backend.Const(fillVal))
      Wrap[Tensor[A]](Adapter.g.reflect("tensor-fill", xs: _*))
    }
  }

  implicit class TensorOps[A: Manifest](xs: Rep[Tensor[A]]) {
    lazy val dims: Seq[Int] = Adapter.g.findDefinition(Unwrap(xs)) match {
      case Some(Node(n, "tensor-new", _:: dims, _)) => dims.map{case Backend.Const(i: Int) => i}
    }
    def checkIdx(idx: Seq[Int]) = {
      assert(dims.length == idx.length, s"Tensor index $idx does not match dimension $dims")
      assert(idx.zip(dims).forall{case (a, b) => a < b}, s"Tensor index $idx is out of bounds for dimension $dims")
    }
    def apply(idx: Int*): Rep[A] = {
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(xs), Backend.Const(idx), Backend.Const(dims))
      Wrap[A](Adapter.g.reflectRead("tensor-apply", unwrapped_xs:_*)(Unwrap(xs)))
    }

    def update(idx: Seq[Int], newVal: A): Unit = {
      checkIdx(idx)
      val mA = Backend.Const(manifest[A])
      val unwrapped_xs: Seq[Backend.Def] = Seq(mA, Unwrap(xs), Backend.Const(idx), Backend.Const(newVal), Backend.Const(dims))
      Wrap[A](Adapter.g.reflectWrite("tensor-update", unwrapped_xs:_*)(Unwrap(xs)))
    }
  }
}

trait BaseGenTensorOps extends DslGenC {
  override def shallow(node: Node) = node match {
    case Node(s, "tensor-new", rhs, eff) => {
      val manifest = rhs.head match {case Const(mani: Manifest[_]) => mani}
      val dims = rhs.tail.map{case Const(i: Int) => i}
      emit(s"malloc(${dims.product} * sizeof(${remap(manifest)}))")
    }
    case Node(s, "tensor-apply", List(manifest, tensor, Const(idx: Seq[Int]), Const(dims: Seq[Int])), eff) => {
      val sizes = dims.scanRight(1)(_ * _).tail
      emit(s"${quote(tensor)}[${idx.zip(sizes).map{case (a, b) => a*b}.sum}]")
    }
    case Node(s, "tensor-update", List(_, tensor, Const(idx: Seq[Int]), newVal, Const(dims: Seq[Int])), _) => {
      val sizes = dims.scanRight(1)(_ * _).tail
      emit(s"${quote(tensor)}[${idx.zip(sizes).map{case (a, b) => a*b}.sum}] = ${quote(newVal)}")
    }
    case n @ Node(s,"P",List(x),_) =>
      emit("""printf("""");
      emit(format(x))
      emit("""\n", """) // Should look like <BEGIN>\n", <END>
      shallow(x);
      emit(")");

    case _ => super.shallow(node)
  }
  def format(x: Def): String = x match {
    case exp: Exp => exp match {
      case s@Sym(_) => typeMap(s).toString match {
        case m if m.contains("scala.lms.tutorial.Tensor[") =>
          "%p"
        case "Float" =>
          "%f"
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
        val tensor = Tensor[Float](Seq(1, 2, 3))
        println(tensor)
        tensor(Seq(0, 1, 2)) = 1
        println(tensor(0, 1, 2))
        println(123)
      }
    }
    dslDriver.eval("5")
  }
}

