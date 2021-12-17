#ifndef INCLUDE_DIM_MPI_CSR
#define INCLUDE_DIM_MPI_CSR

#include <dim/mat/storage_formats/csr.h>

#include <mpi.h>

#include <filesystem>
#include <string>

namespace dim::mpi {

auto load_csr_partial(const std::filesystem::path& path, const std::string& group_name, MPI_Comm communicator)
  -> mat::csr_partial_t<>;
} // namespace dim::mpi

#endif /* INCLUDE_DIM_MPI_CSR */
