#include <dim/mpi/mpi.h>
#include <mpi.h>

namespace dim::mpi {

ctx::ctx(int& argc, char**& argv) { MPI_Init(&argc, &argv); }
ctx::~ctx() { MPI_Finalize(); }
} // namespace dim::mpi