#include <dim/mpi/csr5/edge_sync.h>

#include <dim/mpi/mpi.h>
#include <mpi.h>

namespace dim::mpi::csr5 {

auto edge_sync_t::sync(std::span<double> partial) noexcept -> future_sync_t {
    // send requests to both sides.
    if (_left_sync && _right_sync) {
        return future_sync_t{request_t::active(partial.front(), _left_sync.get()),
                             request_t::active(partial.back(), _right_sync.get())};
    }

    // only left sync our first element overlaps with some node(s) before.
    if (_left_sync)
        return future_sync_t{request_t::active(partial.front(), _left_sync.get()), request_t::inactive()};

    // only right sync, our last element overlaps with some node(s) after.
    if (_right_sync)
        return future_sync_t{request_t::inactive(), request_t::active(partial.back(), _right_sync.get())};

    return future_sync_t{request_t::inactive(), request_t::inactive()};
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

auto edge_sync_t::request_t::await() noexcept -> double {
    ::MPI_Wait(&_request, MPI_STATUS_IGNORE);
    return _to_sync;
}

auto edge_sync_t::future_sync_t::await(std::span<double> result) noexcept -> void {
    if (_left_await.active() && _right_await.active()) {
        auto synced = request_t::await_all(_left_await, _right_await);
        result.front() = synced[0];
        result.back() = synced[1];
        return;
    }

    if (_left_await.active())
        result.front() = _left_await.await();

    if (_right_await.active())
        result.back() = _right_await.await();
}

edge_sync_t::request_t::request_t(active_tag args) : _to_sync{args.to_sync} {
    ::MPI_Iallreduce(&_to_sync, &_to_sync, 1, MPI_DOUBLE, MPI_SUM, args.comm, &_request);
}
} // namespace dim::mpi::csr5