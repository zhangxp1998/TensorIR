package scala.tensor.ir.backend

import lms.core.Backend.{Const, Exp, Node, Sym}
import lms.core.{Backend, Graph, Traverser}
import tensor.ir.AllocationType
import tensor.ir.StagedMemoryAllocator.{Allocation, Deallocation, MemoryEvent}

import scala.collection.mutable

class MemoryRequest(val allocatedTime: Int, var deallocatedTime: Int, var lastUseSym: Sym, val size: Int, val src: Sym, val isCopy: Boolean, val allocType: AllocationType.AllocationType) {
  override def toString: String = s"[$allocatedTime, $deallocatedTime]: $size"
}

object MemoryPlanningTraverser {
  def toEvents(requests: Map[Sym, MemoryRequest]) = {
    val bst = scala.collection.mutable.TreeMap[Int, MemoryEvent]()
    requests.foreach{
      case (key, req) =>
        bst(req.allocatedTime) = Allocation(key.n, req.size)
        bst(req.deallocatedTime) = Deallocation(key.n, req.size, req.lastUseSym)
    }
    bst
  }
}
class MemoryPlanningTraverser extends Traverser {
  var time: Int = 0
  def getTime(): Int = {
    time+=1
    time
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
    MemoryPlanningTraverser.toEvents(requests.toMap)
  }
  def typeSize(mA: Manifest[_]) = mA.toString() match {
    case "Boolean" => 1
    case "Float" => 4
    case "Double" => 8
    case "Int" => 4
  }
  override def traverse(n: Node): Unit = {
    n match {
      case Node(n, "tensor-new", List(Const(mA: Manifest[_]), Const(dims: Seq[Int]), Const(allocType: AllocationType.AllocationType)), _) =>
        requests(n) = new MemoryRequest(getTime(), getTime(), n, dims.product * typeSize(mA), null, false, allocType)
      case Node(n, "tensor-copy", _:: src :: Const(dims: Seq[Int]) :: Const(allocType: AllocationType.AllocationType)::Nil, eff) =>
        src match {
          case exp: Exp => exp match {
            case s@Sym(_) =>
              val size = requests(s).size
              requests(n) = new MemoryRequest(getTime(), getTime(), n, size, s, true, allocType)
              requests(s).deallocatedTime = getTime()
              requests(s).lastUseSym = n
          }
        }
      case Node(n, op, _, eff) =>
        assert(op != "tensor-new", s"$op should be already handled.")
        assert(op != "tensor-copy", s"$op should be already handled.")
        (eff.rkeys ++ eff.wkeys).foreach{
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
    super.traverse(n)
  }
}
