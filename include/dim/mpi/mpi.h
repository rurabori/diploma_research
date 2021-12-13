#pragma once

#include <dim/mpi/comm.h>

#include <mpi.h>

#include <fmt/format.h>

namespace dim::mpi {

struct ctx
{
    ctx(int& argc, char**& argv);
    ctx(const ctx&) = delete;
    ctx(ctx&&) = delete;
    ctx& operator=(const ctx&) = delete;
    ctx& operator=(ctx&&) = delete;
    ~ctx();
};

template<const auto& Fun, typename Ty, typename... Args>
auto query_com(MPI_Comm comm, Args&&... args) {
    Ty result;
    if (const auto status = Fun(comm, std::forward<Args>(args)..., &result); status != MPI_SUCCESS)
        throw std::runtime_error{fmt::format("MPI failed with status code: {}", status)};

    return result;
}

auto rank(MPI_Comm comm = MPI_COMM_WORLD) -> size_t;
auto size(MPI_Comm comm = MPI_COMM_WORLD) -> size_t;

[[nodiscard]] auto split_comm(MPI_Comm parent, int color, int key) -> comm_t;

} // namespace dim::mpi
