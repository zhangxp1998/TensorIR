package scala.lms.tutorial


import lms.core._
import lms.util._
import lms.core.stub._
import lms.core.Backend._
import lms.core.virtualize
import lms.core.utils.time
import lms.macros.{RefinedManifest, SourceContext}

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
  }

  implicit class TensorOps[A: Manifest](xs: Rep[Tensor[A]]) {

  }

}

trait BaseGenTensorOps extends DslGenC {
  override def shallow(node: Node) = node match {
    case Node(s, "tensor-new", rhs, eff) => {
      val manifest = rhs.head match {case Const(mani: Manifest[_]) => mani}
      val dims = rhs.tail.map{case Const(i: Int) => i}
      emit(s"malloc(${dims.product} * sizeof(${remap(manifest)}))")
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
      case s@Sym(_) => typeMap(s) match {
        case _: Manifest[Tensor[_]] =>
          "%p"
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
        println(123)
      }
    }
    dslDriver.eval("5")
  }
}

