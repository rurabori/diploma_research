#include <dim/csr5_mpi/edge_sync.h>

#include <dim/mpi/mpi.h>

namespace dim::csr5_mpi {

namespace {
    auto create_request(double& to_sync, MPI_Comm comm) -> MPI_Request {
        MPI_Request request{};
        ::MPI_Iallreduce(&to_sync, &to_sync, 1, MPI_DOUBLE, MPI_SUM, comm, &request);
        return request;
    }

    auto sync_one(double& to_sync, MPI_Comm comm) -> void {
        auto* request = create_request(to_sync, comm);
        ::MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
} // namespace

auto edge_sync_t::sync(std::span<double> partial) noexcept -> void {
    // send requests to both sides.
    if (_left_sync && _right_sync) {
        MPI_Request requests[2] = {create_request(partial.front(), _left_sync.get()), //
                                   create_request(partial.back(), _right_sync.get())};

        ::MPI_Waitall(std::size(requests), std::data(requests), MPI_STATUSES_IGNORE);
        return;
    }

    // only left sync our first element overlaps with some node(s) before.
    if (_left_sync)
        sync_one(partial.front(), _left_sync.get());

    // only right sync, our last element overlaps with some node(s) after.
    if (_right_sync)
        sync_one(partial.back(), _right_sync.get());
}

auto edge_sync_t::create(size_t left_sync_root, MPI_Comm parent_comm) -> edge_sync_t {
    const auto comm_size = mpi::size(parent_comm);
    const auto node_rank = mpi::rank(parent_comm);

    edge_sync_t result{};

    for (size_t rank = 0; rank < comm_size - 1; ++rank) {
        const auto as_int = static_cast<int>(node_rank);
        if (rank == node_rank) {
            // we always give every process the opportunity to register in this rank.
            result._right_sync = dim::mpi::split_comm(parent_comm, as_int, as_int);
        } else if (rank == left_sync_root) {
            // if we're syncing in this nodes color, we register, else we don't.
            result._left_sync = dim::mpi::split_comm(parent_comm, static_cast<int>(rank), as_int);
        } else {
            (void)dim::mpi::split_comm(parent_comm, MPI_UNDEFINED, MPI_UNDEFINED);
        }
    }

    return result;
}

} // namespace dim::csr5_mpi