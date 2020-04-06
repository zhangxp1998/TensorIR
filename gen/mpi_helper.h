#ifndef __MPI_HELPER_H
#define __MPI_HELPER_H
#include <stdlib.h>
#include <mpi.h>
#include <stdio.h>
#include <algorithm>

static void check_mpi_err(int err) noexcept {
  if (err != MPI_SUCCESS) {
    char err_buffer[1024 * 512];
    int result_len = -1;
    MPI_Error_string(err, err_buffer, &result_len);
    fprintf(stderr, "%s\n", err_buffer);
    abort();
  }
}

static int get_comm_rank(MPI_Comm comm) noexcept {
  int rank = -1;
  int err = MPI_Comm_rank(comm, &rank);
  check_mpi_err(err);
  return rank;
}

static int get_comm_size(MPI_Comm comm) noexcept {
  int size = -1;
  int err = MPI_Comm_size(comm, &size);
  check_mpi_err(err);
  return size;
}

template <typename T>
MPI_Datatype get_mpi_type();

template <>
MPI_Datatype get_mpi_type<float>() {
  return MPI_FLOAT;
}

template <>
MPI_Datatype get_mpi_type<int>() {
  return MPI_INT;
}

template <>
MPI_Datatype get_mpi_type<double>() {
  return MPI_DOUBLE;
}

template <>
MPI_Datatype get_mpi_type<long>() {
  return MPI_LONG;
}

template <>
MPI_Datatype get_mpi_type<char>() {
  return MPI_CHAR;
}


template
<size_t size, typename T>
void MPI_All_average(MPI_Comm comm, const void *send_buf, T *recv_buf, size_t comm_size) {
  
  int err = MPI_Allreduce(send_buf, recv_buf, size, get_mpi_type<T>(), MPI_SUM, comm);
  check_mpi_err(err);
  std::transform(recv_buf, recv_buf+size, recv_buf, [comm_size](T f){ return f/comm_size; });
}

#endif
