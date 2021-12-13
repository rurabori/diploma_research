#ifndef INCLUDE_DIM_CSR5_MPI_RESULT_SYNC
#define INCLUDE_DIM_CSR5_MPI_RESULT_SYNC

#include <dim/csr5_mpi/output_range.h>

#include <mpi.h>

#include <vector>

namespace dim::csr5_mpi {

class result_sync_t
{
    //! @brief the communicator on which the synchronization will take place.
    MPI_Comm _comm;
    //! @brief offsets for synchronizing whole result vector.
    std::vector<int> _recvoffsets;
    //! @brief counts for synchronizing whole result vector.
    std::vector<int> _recvcounts;

    auto add_node(size_t offset, size_t count) -> void {
        _recvoffsets.emplace_back(offset);
        _recvcounts.emplace_back(count);
    }

    explicit result_sync_t(MPI_Comm comm, size_t node_count);

public:
    static auto create(std::span<const output_range_t> all_output_ranges, MPI_Comm comm) -> result_sync_t;

    auto sync(std::span<double> full_result, std::span<const double> partial_result) -> void;
};

} // namespace dim::csr5_mpi

#endif /* INCLUDE_DIM_CSR5_MPI_RESULT_SYNC */
