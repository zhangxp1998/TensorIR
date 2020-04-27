package tensor.ir.backend

import lms.core.Backend
import lms.core.Backend.{Const, Def, Exp, Node, Sym}
import lms.core.stub.DslGenC

trait PrintfCodeGen extends DslGenC {
  override def shallow(n: Backend.Node): Unit = n match {
    case Node(s,"P",List(x),_) =>
      emit("""printf("""")
      emit(format(x))
      emit("""\n", """) // Should look like <BEGIN>\n", <END>
      shallow(x)
      emit(")")
    case _ => super.shallow(n)
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
