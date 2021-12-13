#ifndef INCLUDE_DIM_CSR5_MPI_CSR5_MPI
#define INCLUDE_DIM_CSR5_MPI_CSR5_MPI

#include <dim/csr5_mpi/edge_sync.h>
#include <dim/csr5_mpi/output_range.h>
#include <dim/csr5_mpi/result_sync.h>
#include <dim/mat/storage_formats/csr5.h>

#include <filesystem>
#include <string>

#include <mpi.h>

namespace dim::csr5_mpi {

struct sync_t
{
    edge_sync_t edge_sync;
    result_sync_t result_sync;
};

struct csr5_partial
{
    using matrix_type = mat::csr5<double>;

    MPI_Comm communicator{};
    matrix_type matrix;

    static auto load(const std::filesystem::path& path, const std::string& group_name,
                     MPI_Comm communicator = MPI_COMM_WORLD) -> csr5_partial;

    [[nodiscard]] auto output_range() const noexcept -> output_range_t {
        return output_range_t{.first_row = matrix.first_row_idx(), .last_row = matrix.last_row_idx()};
    }

    [[nodiscard]] auto all_output_ranges() const noexcept -> std::vector<output_range_t>;

    [[nodiscard]] auto make_edge_sync() const noexcept -> edge_sync_t;

    [[nodiscard]] auto make_edge_sync(std::span<const output_range_t> output_ranges) const noexcept -> edge_sync_t;

    [[nodiscard]] auto make_result_sync() const noexcept -> result_sync_t;
    [[nodiscard]] auto make_result_sync(std::span<const output_range_t> output_ranges) const noexcept -> result_sync_t;

    [[nodiscard]] auto make_sync() const noexcept -> sync_t;
    [[nodiscard]] auto make_sync(std::span<const output_range_t> output_ranges) const noexcept -> sync_t;
};

} // namespace dim::csr5_mpi

#endif /* INCLUDE_DIM_CSR5_MPI_CSR5_MPI */
