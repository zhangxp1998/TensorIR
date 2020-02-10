package scala.lms.tutorial

import optimus.optimization._
import optimus.optimization.enums.SolverLib
import optimus.optimization.model.{MPBinaryVar, MPIntVar}

import scala.collection.mutable
import scala.lms.tutorial.StagedMemoryAllocator.{Allocation, Deallocation, MemoryBlock, MemoryEvent}


object MemorySolver {
  def solve(events: Seq[MemoryEvent]): Map[Int, MemoryBlock] = {
    implicit val model = MPModel(SolverLib.oJSolver)
    val maxsize = events.filter(_.isInstanceOf[Allocation]).map{case Allocation(_, size) => size}.sum
    val activeBlocks = new mutable.TreeSet[Int]()
    val blockVars = new mutable.TreeMap[Int, MPIntVar]()
    val allocationSizes: Map[Int, Int] =
      events.filter(_.isInstanceOf[Allocation]).map{case Allocation(id, size) => (id, size)}.toMap[Int, Int]

    val cap = MPIntVar("cap", 0 until maxsize)
    var lower_bound = 0
    var sum = 0
    for (event <- events) {
      event match {
        case Allocation(id, size) =>
          val x = MPIntVar("x"+id, 0 until maxsize)
          for ((y_id, y_var: MPIntVar) <- activeBlocks.map(x => (x, blockVars(x)))) {
            val b1 = MPBinaryVar(s"b_${id}_$y_id")
            val b2 = MPBinaryVar(s"b_${y_id}_$id")
            add(x + size <:= y_var + maxsize * b1)
            add((y_var + allocationSizes(y_id)) <:= x + maxsize * b2)
            add(b1 + b2 <:= 1)
          }
          add(x + size <:= cap)
          activeBlocks += id
          blockVars(id) = x
          sum += size
          lower_bound = math.max(lower_bound, sum)
        case Deallocation(id, size, _) =>
          activeBlocks -= id
          sum -= size
        case _ =>
      }
    }

    minimize(cap)
    val allocationPlan = new mutable.TreeMap[Int, MemoryBlock]()
    try {
      start()
      println(s"objective: ${math.round(objectiveValue)} lower bound: $lower_bound")


      for ((id, blk_var) <- blockVars) {
        println(s"$id = ${blk_var.value}")
        allocationPlan(id) = MemoryBlock(math.round(blk_var.value.get).toInt, allocationSizes(id))
      }
    } finally {
      release()
    }
    allocationPlan.toMap
  }

  def main(args: Array[String]): Unit = {
//    solve()
  }
}