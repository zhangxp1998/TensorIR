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
  override def traverse(n: Node): Unit = {
    n match {
      case Node(n, "tensor-new", List(mA, Backend.Const(dims: Seq[Int]), Const(allocType: AllocationType.AllocationType)), _) =>
        requests(n) = new MemoryRequest(getTime(), getTime(), n, dims.product, null, false, allocType)
      case Node(n, "tensor-copy", _:: src :: dims :: Const(allocType: AllocationType.AllocationType)::Nil, eff) =>
        val size = (dims match {case Backend.Const(seq: Seq[Int]) => seq}).product
        src match {
          case exp: Exp => exp match {
            case s@Sym(_) =>
              requests(n) = new MemoryRequest(getTime(), getTime(), n, size, s, true, allocType)
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
