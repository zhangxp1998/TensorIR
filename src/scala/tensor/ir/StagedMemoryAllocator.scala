package tensor.ir

import lms.core.Backend.Sym

import scala.collection.mutable

object StagedMemoryAllocator {
  class MemoryEvent {

  }
  case class Allocation(id: Int, size: Int) extends MemoryEvent
  case class Deallocation(id: Int, size: Int, afterSym: Sym) extends MemoryEvent

  case class MemoryBlock(begin: Int, size: Int)

  def allocate(events: Seq[MemoryEvent]): Map[Int, MemoryBlock] = {
    // From size to memory block
    val freelist = new mutable.TreeMap[Int, mutable.Set[MemoryBlock]]()
    val maxsize = events.filter(_.isInstanceOf[Allocation]).map{case Allocation(_, size) => size.toLong}.sum

    freelist.put(maxsize.toInt, mutable.Set(MemoryBlock(0, maxsize.toInt)))

    var mem_used = 0;
    var min_mem = 0;

    val allocationPlan = new mutable.TreeMap[Int, MemoryBlock]()
    events.foreach {
      case Allocation(id, size) =>
        val minSize = freelist.keys.find(s => s >= size).get
        val freeblocks = freelist(minSize)
        val block = freeblocks.head
        freeblocks.remove(block)
        if (freeblocks.isEmpty) freelist.remove(minSize)
        if (block.size == size) {
          allocationPlan(id) = block
        } else {
          allocationPlan(id) = MemoryBlock(block.begin, size)
          val remain = MemoryBlock(block.begin + size, block.size - size)
          val list = freelist.getOrElseUpdate(remain.size, mutable.Set())
          list += remain
        }
        mem_used += size
        assert(mem_used + freelist.values.flatMap(_.map(_.size)).sum == maxsize)
        min_mem = Math.max(mem_used, min_mem)

      case Deallocation(id, size, afterSym) =>
        var block = allocationPlan(id)
        freelist.foreach { case (blockSize, blocks) =>
          blocks.find(_.begin == block.begin + block.size) match {
            case Some(rightNeighbor) =>
              blocks -= rightNeighbor
              if (blocks.isEmpty) {
                freelist.remove(blockSize)
              }
              block = MemoryBlock(block.begin, block.size + rightNeighbor.size)
            case None =>
          }
          blocks.find(b => b.begin + b.size == block.begin) match {
            case Some(leftNeighbor) =>
              blocks -= leftNeighbor
              if (blocks.isEmpty) {
                freelist.remove(blockSize)
              }
              block = MemoryBlock(leftNeighbor.begin, block.size + leftNeighbor.size)
            case None =>
          }
        }
        freelist.getOrElseUpdate(block.size, mutable.Set()).add(block)
        mem_used -= size
        assert(mem_used + freelist.values.flatMap(_.map(_.size)).sum == maxsize)
    }
    val lastBlk = allocationPlan.values.maxBy(b => b.begin + b.size)
    println(s"Optimal: $min_mem Actual: ${lastBlk.begin + lastBlk.size}")
    allocationPlan.toMap
  }
}
