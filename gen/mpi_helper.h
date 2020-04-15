#ifndef __MPI_HELPER_H
#define __MPI_HELPER_H
#include <stdlib.h>
#include <mpi.h>
#include <stdio.h>
#include <algorithm>
#include <chrono>

void check_mpi_err(int err) noexcept;

int get_comm_rank(MPI_Comm comm) noexcept;

int get_comm_size(MPI_Comm comm) noexcept;

template <typename T>
[[maybe_unused]] static MPI_Datatype get_mpi_type() noexcept;

template <>
MPI_Datatype get_mpi_type<float>() noexcept {
  return MPI_FLOAT;
}

template <>
MPI_Datatype get_mpi_type<int>() noexcept {
  return MPI_INT;
}

template <>
MPI_Datatype get_mpi_type<double>() noexcept {
  return MPI_DOUBLE;
}

template <>
MPI_Datatype get_mpi_type<long>() noexcept {
  return MPI_LONG;
}

template <>
MPI_Datatype get_mpi_type<char>() noexcept {
  return MPI_CHAR;
}


extern std::chrono::milliseconds mpi_duration;

template
<size_t size, typename T>
void MPI_All_average(MPI_Comm comm, const void *send_buf, T *recv_buf, size_t comm_size) {
  if (comm_size == 1) {
    return;
  }
  using namespace std::chrono;
  auto start = system_clock::now();
  int err = MPI_Allreduce(send_buf, recv_buf, size, get_mpi_type<T>(), MPI_SUM, comm);
  check_mpi_err(err);
  std::transform(recv_buf, recv_buf+size, recv_buf, [comm_size](T f){ return f/comm_size; });
  auto end = system_clock::now();
  mpi_duration += duration_cast<decltype(mpi_duration)>(end - start);
}

#endif
