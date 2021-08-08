#include <dim/mpi/mpi.h>

namespace dim::mpi {

ctx::ctx(int& argc, char**& argv) { MPI::Init(argc, argv); }
ctx::~ctx() { MPI::Finalize(); }
} // namespace dim::mpi