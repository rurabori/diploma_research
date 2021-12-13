#include <dim/csr5_mpi/output_range.h>

namespace dim::csr5_mpi {

auto output_range_t::syncs_downto(std::span<const output_range_t> all_output_ranges, size_t current) -> size_t {
    const auto base = all_output_ranges[current];

    while (current != 0 && base.is_continuation_of(all_output_ranges[current - 1]))
        --current;

    return current;
}

} // namespace dim::csr5_mpi