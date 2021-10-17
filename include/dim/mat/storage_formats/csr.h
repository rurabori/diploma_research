#ifndef INCLUDE_DIM_MAT_STORAGE_FORMATS_CSR
#define INCLUDE_DIM_MAT_STORAGE_FORMATS_CSR

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include <dim/mat/storage_formats/coo.h>
#include <dim/memory/aligned_allocator.h>

namespace dim::mat {

template<typename ValueType, template<typename> typename StorageContainer = cache_aligned_vector>
struct csr
{
    using values_t = StorageContainer<ValueType>;
    using indices_t = StorageContainer<uint32_t>;

    dimensions_t dimensions{};
    values_t values;
    indices_t row_start_offsets;
    indices_t col_indices;

    csr(dimensions_t dimensions, size_t non_zero_count)
      : dimensions{dimensions},
        values(non_zero_count),
        row_start_offsets(dimensions.rows + 1),
        col_indices(non_zero_count) {}

    csr(dimensions_t dimensions, values_t values, indices_t row_start_offsets, indices_t col_indices)
      : dimensions{dimensions},
        values{std::move(values)},
        row_start_offsets{std::move(row_start_offsets)},
        col_indices{std::move(col_indices)} {}

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
        std::copy(csr_row_counter.begin(), csr_row_counter.end(), retval.row_start_offsets.begin());

        // reuse the csr_row_counter as offset counter for rows.
        std::fill(csr_row_counter.begin(), csr_row_counter.end(), 0);
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

#endif /* INCLUDE_DIM_MAT_STORAGE_FORMATS_CSR */
