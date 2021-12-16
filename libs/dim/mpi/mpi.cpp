#include <dim/mpi/mpi.h>
#include <mpi.h>

namespace dim::mpi {

ctx::ctx(int& argc, char**& argv) { MPI_Init(&argc, &argv); }
ctx::~ctx() { MPI_Finalize(); }

auto rank(MPI_Comm comm) -> size_t { return static_cast<size_t>(query_com<::MPI_Comm_rank, int>(comm)); }
auto size(MPI_Comm comm) -> size_t { return static_cast<size_t>(query_com<::MPI_Comm_size, int>(comm)); }

auto split_comm(MPI_Comm parent, int color, int key) -> comm_t {
    auto* tmp = query_com<::MPI_Comm_split, MPI_Comm>(parent, color, key);
    return comm_t{tmp == MPI_COMM_NULL ? nullptr : tmp};
}
} // namespace dim::mpi