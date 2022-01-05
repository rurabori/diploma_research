#ifndef INCLUDE_DIM_MPI_CSR5
#define INCLUDE_DIM_MPI_CSR5

#include <dim/mat/storage_formats/csr5.h>
#include <dim/mpi/csr5/edge_sync.h>
#include <dim/mpi/csr5/output_range.h>
#include <dim/mpi/csr5/result_sync.h>

#include <filesystem>
#include <string>

#include <mpi.h>

namespace dim::mpi::csr5 {

struct sync_t
{
    edge_sync_t edge_sync;
    result_sync_t result_sync;
};

class csr5_partial
{
public:
    using matrix_type = mat::csr5<double>;

private:
    MPI_Comm _communicator{};
    matrix_type _matrix;

    [[nodiscard]] auto make_edge_sync() const noexcept -> edge_sync_t;

    [[nodiscard]] auto make_edge_sync(std::span<const output_range_t> output_ranges) const noexcept -> edge_sync_t;

    [[nodiscard]] auto make_result_sync() const noexcept -> result_sync_t;
    [[nodiscard]] auto make_result_sync(std::span<const output_range_t> output_ranges) const noexcept -> result_sync_t;
    [[nodiscard]] auto make_sync(std::span<const output_range_t> output_ranges) const noexcept -> sync_t;

    csr5_partial(MPI_Comm comm, matrix_type&& mat) : _communicator{comm}, _matrix{std::move(mat)} {}

public:
    static auto load(const std::filesystem::path& path, const std::string& group_name,
                     MPI_Comm communicator = MPI_COMM_WORLD) -> csr5_partial;

    [[nodiscard]] auto matrix() const noexcept -> const matrix_type& { return _matrix; }

    template<typename... Args>
    auto spmv_partial(Args&&... args) const noexcept -> void {
        _matrix.spmv<matrix_type::spmv_strategy::partial>(std::forward<Args>(args)...);
    }

    [[nodiscard]] auto make_sync() const noexcept -> sync_t;

    [[nodiscard]] auto output_range() const noexcept -> output_range_t {
        return output_range_t{.first_row = _matrix.first_row_idx(), .last_row = _matrix.last_row_idx()};
    }

    [[nodiscard]] auto all_output_ranges() const noexcept -> std::vector<output_range_t>;
};

} // namespace dim::mpi::csr5

#endif /* INCLUDE_DIM_MPI_CSR5 */
