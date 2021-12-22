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

    class request_t
    {
        double result{};
        MPI_Request request{MPI_REQUEST_NULL};

    public:
        static auto create(double to_sync, MPI_Comm comm) -> request_t;

        static auto inactive() -> request_t { return {}; }

        [[nodiscard]] auto active() const noexcept -> bool { return request != MPI_REQUEST_NULL; }

        auto await() noexcept -> double;

        static auto await_all(std::same_as<request_t> auto&... requests) -> std::array<double, sizeof...(requests)> {
            MPI_Request handles[] = {requests.request...};
            ::MPI_Waitall(std::size(handles), std::data(handles), MPI_STATUSES_IGNORE);
            return {requests.result...};
        }
    };

    class future_sync_t
    {
        request_t _left_await;
        request_t _right_await;

    public:
        future_sync_t(request_t left, request_t right) : _left_await{left}, _right_await{right} {}

        auto await(std::span<double> result) noexcept -> void;
    };

public:
    [[nodiscard]] auto sync(std::span<double> partial) noexcept -> future_sync_t;

    static auto create(size_t left_sync_root, MPI_Comm parent_comm = MPI_COMM_WORLD) -> edge_sync_t;
};

} // namespace dim::csr5_mpi

#endif /* INCLUDE_DIM_CSR5_MPI_EDGE_SYNC */
