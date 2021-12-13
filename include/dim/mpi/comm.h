#ifndef INCLUDE_DIM_MPI_COMM
#define INCLUDE_DIM_MPI_COMM

#include <mpi.h>

#include <memory>

namespace dim::mpi {

struct comm_deleter
{
    auto operator()(MPI_Comm comm) const noexcept -> void {
        if (comm)
            ::MPI_Comm_disconnect(&comm);
    }
};
using comm_t = std::unique_ptr<std::remove_pointer_t<MPI_Comm>, comm_deleter>;

} // namespace dim::mpi

#endif /* INCLUDE_DIM_MPI_COMM */
