package tensor.ir

import java.io.PrintWriter

import optimus.optimization.enums.SolverLib
import optimus.optimization.model.{MPBinaryVar, MPIntVar}
import optimus.optimization._
import tensor.ir.StagedMemoryAllocator.{Allocation, Deallocation, MemoryBlock, MemoryEvent}

import scala.collection.mutable

object MemorySolver {
  def allocate(events: Seq[MemoryEvent]): Unit /*Map[Int, MemoryBlock]*/ = {
//    val maxsize = events.filter(_.isInstanceOf[Allocation]).map{case Allocation(_, size) => size}.sum
    val activeBlocks = new mutable.TreeSet[Int]()
    val allocationSizes: Map[Int, Int] =
      events.filter(_.isInstanceOf[Allocation]).map{case Allocation(id, size) => (id, size)}.toMap[Int, Int]

    var lower_bound = 0
    var sum = 0
    val fileName = "memory.z3"
    val writer = new PrintWriter(fileName)
    val cap = "cap"
    try {
      writer.println("(define-fun max ((x Int) (y Int)) Int\n  (ite (< x y) y x))")
      writer.println(s"(declare-const $cap Int)")
      events.foreach {
        case Allocation(id, size) =>
          val x = "x_" + id
          writer.println(s"(declare-const $x Int)")
          writer.println(s"(assert (>= $x 0))")
          writer.println(s"(assert (>= $cap (+ $x $size)))")
          activeBlocks.foreach(y_id => {
            val y = "x_" + y_id
            writer.println(s"(assert (or (<= (+ $x $size) $y) (<= (+ $y ${allocationSizes(y_id)}) $x) ))")
          })
          activeBlocks += id
          sum += size
          lower_bound = math.max(lower_bound, sum)
        case Deallocation(id, size, afterSym) =>
          activeBlocks -= id
          sum -= size
      }
      writer.println(s"(minimize $cap)")
      writer.println("(check-sat)")
      writer.println("(get-model)")
      writer.println("(get-objectives)")
      writer.flush()
      import scala.sys.process._
      (s"z3 $fileName": ProcessBuilder).lineStream.foreach(println)
    } finally {
      writer.close()
    }
  }
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
