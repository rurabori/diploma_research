#ifndef INCLUDE_DIM_MAT_STORAGE_FORMATS
#define INCLUDE_DIM_MAT_STORAGE_FORMATS

#include <cstddef>
#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>

#include <dim/memory/aligned_allocator.h>



namespace dim::mat {

struct dimensions_t
{
    uint32_t rows{};
    uint32_t cols{};
};

template<typename Ty>
using cache_aligned_vector = std::vector<Ty, dim::memory::cache_aligned_allocator_t<Ty>>;

template<typename ValueType, template<typename> typename StorageContainer = cache_aligned_vector>
struct coo
{
    dimensions_t dimensions{};
    StorageContainer<ValueType> values;
    StorageContainer<uint32_t> row_indices;
    StorageContainer<uint32_t> col_indices;
    bool symmetric{false};

    coo(dimensions_t dimensions, size_t non_zero_count, bool symmetric)
      : dimensions{dimensions},
        values(non_zero_count),
        row_indices(non_zero_count),
        col_indices(non_zero_count),
        symmetric{symmetric} {}

    template<std::invocable<ValueType, uint32_t, uint32_t> Callable>
    void iterate(Callable&& callable) const noexcept {
        for (size_t i = 0; i < values.size(); ++i) {
            const auto row = row_indices[i];
            const auto col = col_indices[i];

            // send the values in.
            callable(values[i], row, col);

            // for symmetric matrices, we also need to invoke with the reverse.
            if (symmetric && row != col)
                callable(values[i], col, row);
        }
    }
};

template<typename ValueType, template<typename> typename StorageContainer = cache_aligned_vector>
struct csr
{
    dimensions_t dimensions{};
    StorageContainer<ValueType> values;
    StorageContainer<uint32_t> row_start_offsets;
    StorageContainer<uint32_t> col_indices;

    csr(dimensions_t dimensions, size_t non_zero_count)
      : dimensions{dimensions},
        values(non_zero_count),
        row_start_offsets(dimensions.rows + 1),
        col_indices(non_zero_count) {}

    template<template<typename> typename StorageTy>
    static csr from_coo(const coo<ValueType, StorageTy>& coo) {
        // count occurences of rows.
        std::vector<uint32_t> csr_row_counter(coo.dimensions.rows + 1, 0);

        coo.iterate([&](auto /*value*/, auto row, auto /*col*/) { ++csr_row_counter[row]; });

        // prefix scan to get the starting indices of cols in a row.
        std::exclusive_scan(csr_row_counter.begin(), csr_row_counter.end(), csr_row_counter.begin(), 0);

        // the number of actually non-0 elements is the last element in the offset array.
        csr retval{coo.dimensions, csr_row_counter.back()};
        // copy the offset array into the struct.
        std::ranges::copy(csr_row_counter, retval.row_start_offsets.begin());

        // reuse the csr_row_counter as offset counter for rows.
        std::ranges::fill(csr_row_counter, 0);
        coo.iterate([&](auto value, auto row, auto col) {
            const auto row_start = retval.row_start_offsets[row];
            const auto offset = row_start + csr_row_counter[row]++;

            retval.col_indices[offset] = col;
            retval.values[offset] = value;
        });

        return retval;
    }
};

} // namespace dim::mat

#endif /* INCLUDE_DIM_MAT_STORAGE_FORMATS */
