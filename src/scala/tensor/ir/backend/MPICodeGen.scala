package tensor.ir.backend

import lms.core.Backend
import lms.core.Backend.Node
import lms.core.stub.DslGenC
import tensor.ir.RandomOpsCodegen

trait MPICodeGen extends DslGenC {

  def mpi_init() = {
    registerInit("MPI") {
      emitln("MPI_Init(pargc, pargv);")
    }
    registerHeader("\"mpi_helper.h\"")
  }
  override def remap(m: Manifest[_]): String = m.toString match {
    case s if s.contains("MPI_Comm") => "MPI_Comm"
    case _ => super.remap(m)
  }
  override def shallow(n: Backend.Node): Unit = n match {
    case Node(s, "MPI_Comm_rank", List(comm), _) =>
      mpi_init()
      emit("get_comm_rank(")
      shallow(comm)
      emit(")")
    case Node(s, "MPI_Comm_size", List(comm), _) =>
      mpi_init()
      emit("get_comm_size(")
      shallow(comm)
      emit(")")
    case Node(s, "MPI_COMM_WORLD", List(), _) =>
      mpi_init()
      emit("MPI_COMM_WORLD")
    case Node(s, "MPI_All_average", List(comm, data, Backend.Const(data_size: Int), comm_size), _) =>
      mpi_init()
      emit(s"MPI_All_average<$data_size>(")
      shallow(comm)
      emit(", MPI_IN_PLACE, ")
      shallow(data)
      emit(", ")
      shallow(comm_size)
      emit(")")
    case _ => super.shallow(n)
  }
}
