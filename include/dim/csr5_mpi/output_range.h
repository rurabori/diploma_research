#ifndef INCLUDE_DIM_CSR5_MPI_OUTPUT_RANGE
#define INCLUDE_DIM_CSR5_MPI_OUTPUT_RANGE

#include <cstdint>
#include <span>

namespace dim::csr5_mpi {

//! @brief output range for partial matrix, just a pair of first and last row the partial matrix outputs to.
struct output_range_t
{
    uint32_t first_row;
    uint32_t last_row;

    /**
     * @brief Checks if this output_range is continuation of [other]. In other words, if their first and last elements
     * respectively overlap.
     */
    [[nodiscard]] auto is_continuation_of(const output_range_t& other) const noexcept -> bool {
        return other.last_row == first_row;
    }

    /**
     * @brief Returns the number of elements ins this output_range.
     */
    [[nodiscard]] auto count() const noexcept -> uint32_t { return last_row - first_row + 1; }

    /**
     * @brief Returns the index of first output_range that outputs to the all_output_ranges[current].first_row.
     *
     * @param all_output_ranges output ranges for all nodes.
     * @param current index of the output_range to sync.
     * @return size_t index of first output_range that outputs to the all_output_ranges[current].first_row.
     */
    static auto syncs_downto(std::span<const output_range_t> all_output_ranges, size_t current) -> size_t;
};

} // namespace dim::csr5_mpi

#endif /* INCLUDE_DIM_CSR5_MPI_OUTPUT_RANGE */
