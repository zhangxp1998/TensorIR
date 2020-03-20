package scala.tensor.ir.backend

import lms.core.Backend.{Block, Const, Exp, Node, Sym}
import lms.core.Transformer
import lms.core.stub.Adapter
import lms.core.stub.Adapter.typeMap
import tensor.ir.StagedMemoryAllocator.MemoryBlock

import scala.collection.mutable


class CPUMemoryPlanningTransformer(val allocationPlan: Map[Int, MemoryBlock], val reusedSyms: Map[Sym, Sym]) extends Transformer {
  g = Adapter.mkGraphBuilder()
  val totalMemory: Long = {
    if (allocationPlan.isEmpty) 0 else {
      val maxBlock = allocationPlan.values.maxBy(_.begin)
      maxBlock.begin + maxBlock.size
    }
  }

  lazy val newTypeMap: mutable.Map[Exp, Manifest[_]] = {
    val newMap: mutable.Map[Exp, Manifest[_]] = new mutable.HashMap[Exp, Manifest[_]]
    symMap.foreach{
      case (before, after) if typeMap.contains(before) =>
        newMap(after) = typeMap(before)
      case _ =>
    }
    (typeMap -- symMap.values).foreach{
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
    case Node(s, "tensor-new", List(mA, dims, allocType), eff) =>
      val exp = g.reflectEffect("heap-offset", mA, Const(allocationPlan(s.n)))(
        eff.rkeys.map(transform).toSeq: _*
      )(
        eff.wkeys.map(transform).toSeq: _*
      )
      symMap(s) = exp
      exp
    case Node(s, "tensor-copy", List(mA, tensor, dims, allocType), eff) =>
      val arg = tensor match {
        case b @ Block(_,_,_,_) =>
          transform(b)
        case s : Exp =>
          transform(s)
        case a =>
          a
      }
      val exp = if (!allocationPlan.contains(s.n)) {
        assert(reusedSyms.contains(s))
        val src = getSrc(s)
        val exp = g.reflectEffect("heap-offset", mA, Const(allocationPlan(src.n)), arg)(
          eff.rkeys.map(transform).toSeq: _*
        )(
          eff.wkeys.map(transform).toSeq: _*
        )
        exp
      } else {
        val exp = g.reflectEffect("heap-offset-copy", mA, arg, Const(allocationPlan(s.n)), dims)(
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
