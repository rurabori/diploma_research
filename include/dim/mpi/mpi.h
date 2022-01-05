#pragma once

#include <dim/mpi/comm.h>

#include <mpi.h>

#include <fmt/format.h>

namespace dim::mpi {

/**
 * @brief a RAII wrapper for MPI initialization scheme.
 */
struct ctx
{
    /**
     * @brief Initializes a MPI context, i.e. calls MPI_Init.
     */
    ctx(int& argc, char**& argv);
    ctx(const ctx&) = delete;
    ctx(ctx&&) = delete;
    ctx& operator=(const ctx&) = delete;
    ctx& operator=(ctx&&) = delete;
    /**
     * @brief Tears the MPI context down, i.e. calls MPI_Finalize.
     */
    ~ctx();
};

/**
 * @brief Queries the communicator in [comm] with the [Fun] passed in.
 * for example query_comm<MPI_Comm_rank, int>(comm);
 *
 * @tparam Fun The actual function to call.
 * @tparam Ty the type of result.
 */
template<const auto& Fun, typename Ty, typename... Args>
auto query_com(MPI_Comm comm, Args&&... args) {
    Ty result;
    if (const auto status = Fun(comm, std::forward<Args>(args)..., &result); status != MPI_SUCCESS)
        throw std::runtime_error{fmt::format("MPI failed with status code: {}", status)};

    return result;
}

//! @see MPI_Comm_rank
auto rank(MPI_Comm comm = MPI_COMM_WORLD) -> size_t;

//! @see MPI_Comm_size
auto size(MPI_Comm comm = MPI_COMM_WORLD) -> size_t;

//! @see MPI_Comm_split
[[nodiscard]] auto split(MPI_Comm parent, int color, int key) -> comm_t;

} // namespace dim::mpi
