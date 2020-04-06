package tensor.ir

import lms.core.Backend
import lms.core.stub.{Adapter, Base}
import scala.collection.mutable

// A trait that maps to MPI_Comm in C code.
trait MPI_Comm {

}

trait MPI extends Base {
  object MPI {
    val MPI_COMM_WORLD: Rep[MPI_Comm] = get_comm_world()
    val sizeMap = mutable.HashMap[Rep[MPI_Comm], Rep[Int]]()
    def get_comm_world(): Rep[MPI_Comm] = {
      Wrap[MPI_Comm](Adapter.g.reflect("MPI_COMM_WORLD"))
    }
    def comm_rank(comm: Rep[MPI_Comm]): Rep[Int] = {
      Wrap[Int](Adapter.g.reflect("MPI_Comm_rank", Unwrap(comm)))
    }
    def comm_size(comm: Rep[MPI_Comm]): Rep[Int] = {
      Wrap[Int](Adapter.g.reflect("MPI_Comm_size", Unwrap(comm)))
    }
    def All_average[T: Manifest: Ordering](comm: Rep[MPI_Comm], data: Rep[Array[T]], size: Int): Unit = {
      val unwrapped = Unwrap(data)
      val num_procs = sizeMap.getOrElseUpdate(comm, comm_size(comm))
      Adapter.g.reflectEffect("MPI_All_average", Unwrap(comm), unwrapped, Backend.Const(size), Unwrap(num_procs))(unwrapped)(unwrapped)
    }
  }
}
