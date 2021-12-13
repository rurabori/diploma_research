#include <dim/csr5_mpi/result_sync.h>

#include <mpi.h>

namespace dim::csr5_mpi {

result_sync_t::result_sync_t(MPI_Comm comm, size_t node_count) : _comm{comm} {
    _recvcounts.reserve(node_count);
    _recvoffsets.reserve(node_count);
}

auto result_sync_t::create(std::span<const output_range_t> all_output_ranges, MPI_Comm comm) -> result_sync_t {
    auto result = result_sync_t{comm, all_output_ranges.size()};

    result.add_node(all_output_ranges[0].first_row, all_output_ranges[0].count());

    for (size_t idx = 1; idx < all_output_ranges.size(); ++idx) {
        const auto& current = all_output_ranges[idx];
        const bool overlap = current.is_continuation_of(all_output_ranges[idx - 1]);

        result.add_node(current.first_row + overlap, current.count() - overlap);
    }

    return result;
}

auto result_sync_t::sync(std::span<double> full_result, std::span<const double> partial_result) -> void {
    ::MPI_Allgatherv(partial_result.data(), static_cast<int>(partial_result.size()), MPI_DOUBLE, full_result.data(),
                     _recvcounts.data(), _recvoffsets.data(), MPI_DOUBLE, _comm);
}

} // namespace dim::csr5_mpi