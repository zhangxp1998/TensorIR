package scala.lms.tutorial

import java.util

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
    val maxsize = events.filter(_.isInstanceOf[Allocation]).map{case Allocation(_, size) => size}.sum

    freelist.put(maxsize, mutable.Set(MemoryBlock(0, maxsize)))

    val allocationPlan = new mutable.TreeMap[Int, MemoryBlock]()
    events.foreach {
      case Allocation(id, size) =>
        val minSize = freelist.keys.find(s => s >= size).get
        val freeblocks = freelist(minSize)
        val block = freeblocks.head
        if (block.size == size) {
          allocationPlan(id) = block
          freelist(minSize) = freeblocks.tail
        } else {
          allocationPlan(id) = MemoryBlock(block.begin, size)
          val remain = MemoryBlock(block.begin + size, block.size - size)
          val list = freelist.getOrElseUpdate(remain.size, mutable.Set())
          list -= block
          list += remain
        }

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
    }
    allocationPlan.toMap
  }
}