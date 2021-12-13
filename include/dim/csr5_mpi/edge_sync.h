#ifndef INCLUDE_DIM_CSR5_MPI_EDGE_SYNC
#define INCLUDE_DIM_CSR5_MPI_EDGE_SYNC

#include <dim/mpi/comm.h>

#include <mpi.h>
#include <span>

namespace dim::csr5_mpi {

class edge_sync_t
{
    //! @brief communicator on which this node synchronizes to the left.
    mpi::comm_t _left_sync;
    //! @brief communicator on which all nodes to the right of this node sync (for them this is left_sync).
    mpi::comm_t _right_sync;

public:
    auto sync(std::span<double> partial) noexcept -> void;

    static auto create(size_t left_sync_root, MPI_Comm parent_comm = MPI_COMM_WORLD) -> edge_sync_t;
};

} // namespace dim::csr5_mpi

#endif /* INCLUDE_DIM_CSR5_MPI_EDGE_SYNC */
