#include "mpi_helper.h"

std::chrono::milliseconds mpi_duration{};

void check_mpi_err(int err) noexcept {
  if (err != MPI_SUCCESS) {
    char err_buffer[1024 * 512];
    int result_len = -1;
    MPI_Error_string(err, err_buffer, &result_len);
    fprintf(stderr, "%s\n", err_buffer);
    abort();
  }
}

int get_comm_rank(MPI_Comm comm) noexcept {
  int rank = -1;
  int err = MPI_Comm_rank(comm, &rank);
  check_mpi_err(err);
  return rank;
}

int get_comm_size(MPI_Comm comm) noexcept {
  int size = -1;
  int err = MPI_Comm_size(comm, &size);
  check_mpi_err(err);
  return size;
}
