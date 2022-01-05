#ifndef INCLUDE_DIM_MPI_CSR5_EDGE_SYNC
#define INCLUDE_DIM_MPI_CSR5_EDGE_SYNC

#include <dim/mpi/comm.h>

#include <mpi.h>
#include <span>

namespace dim::mpi::csr5 {

class edge_sync_t
{
    //! @brief communicator on which this node synchronizes to the left.
    mpi::comm_t _left_sync;
    //! @brief communicator on which all nodes to the right of this node sync (for them this is left_sync).
    mpi::comm_t _right_sync;

    class request_t
    {
        struct inactive_tag
        {};
        struct active_tag
        {
            double to_sync;
            MPI_Comm comm;
        };

        double _to_sync{};
        MPI_Request _request{MPI_REQUEST_NULL};

    public:
        static constexpr auto inactive() noexcept -> inactive_tag { return {}; }

        static constexpr auto active(double to_sync, MPI_Comm comm) noexcept -> active_tag {
            return {.to_sync = to_sync, .comm = comm};
        }

        explicit request_t(inactive_tag /*unused*/) {}
        explicit request_t(active_tag args);

        request_t() = delete;
        request_t(const request_t&) = delete;
        request_t(request_t&&) = delete;
        request_t& operator=(const request_t&) = delete;
        request_t& operator=(request_t&&) = delete;
        ~request_t() = default;

        [[nodiscard]] auto active() const noexcept -> bool { return _request != MPI_REQUEST_NULL; }

        auto await() noexcept -> double;

        static auto await_all(std::same_as<request_t> auto&... requests) -> std::array<double, sizeof...(requests)> {
            MPI_Request handles[] = {requests._request...};
            ::MPI_Waitall(std::size(handles), std::data(handles), MPI_STATUSES_IGNORE);
            return {requests._to_sync...};
        }
    };

    class future_sync_t
    {
        request_t _left_await;
        request_t _right_await;

    public:
        future_sync_t(auto&& left, auto&& right)
          : _left_await{std::forward<decltype(left)>(left)}, _right_await{std::forward<decltype(right)>(right)} {}

        auto await(std::span<double> result) noexcept -> void;
    };

public:
    [[nodiscard]] auto sync(std::span<double> partial) noexcept -> future_sync_t;

    static auto create(size_t left_sync_root, MPI_Comm parent_comm = MPI_COMM_WORLD) -> edge_sync_t;
};

} // namespace dim::mpi::csr5

#endif /* INCLUDE_DIM_MPI_CSR5_EDGE_SYNC */
