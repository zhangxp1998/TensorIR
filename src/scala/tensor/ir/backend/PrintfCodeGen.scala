package tensor.ir.backend

import lms.core.Backend
import lms.core.Backend.{Const, Def, Exp, Node, Sym}
import lms.core.stub.DslGenC

trait PrintfCodeGen extends DslGenC {
  def emitBeginEnd(data: Def, begin: Def, end: Def): Unit = {
    shallow(data)
    if (begin != Const(0)) {
      emit("+")
      shallow(begin)
    }
    emit(", ")
    shallow(data)
    emit("+")
    shallow(end)
  }
  override def shallow(n: Backend.Node): Unit = n match {
    case Node(s,"P", values ,_) =>
      emit("""printf("""")
      values.zipWithIndex.foreach{ case (x: Def, idx: Int) =>
        emit(format(x))
        if (idx != values.length - 1) {
          emit(" ")
        }
      }
      emit("""\n", """) // Should look like <BEGIN>\n", <END>
      values.zipWithIndex.foreach{ case (x: Def, idx: Int) =>
        shallow(x)
        if (idx != values.length - 1) {
          emit(",")
        }
      }
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
