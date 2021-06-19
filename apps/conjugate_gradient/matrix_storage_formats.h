#ifndef APPS_CONJUGATE_GRADIENT_MATRIX_STORAGE_FORMATS
#define APPS_CONJUGATE_GRADIENT_MATRIX_STORAGE_FORMATS

#include "cache_aligned_allocator.h"
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <vector>

namespace cg {

struct dimensions_t
{
    uint32_t rows{};
    uint32_t cols{};
};

namespace matrix_storage_formats {

    template<typename Ty>
    using cache_aligned_vector = std::vector<Ty, cache_aligned_allocator_t<Ty>>;

    template<typename ValueType, template<typename> typename StorageContainer = cache_aligned_vector>
    struct coo
    {
        dimensions_t dimensions{};
        StorageContainer<ValueType> values;
        StorageContainer<uint32_t> row_indices;
        StorageContainer<uint32_t> col_indices;

        coo(dimensions_t dimensions, size_t non_zero_count)
          : dimensions{dimensions}, values(non_zero_count), row_indices(non_zero_count), col_indices(non_zero_count) {}
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
            csr retval{coo.dimensions, coo.values.size()};

            // count occurences of rows.
            std::vector<uint> csr_row_counter(coo.dimensions.rows + 1, 0);
            for (auto row_id : coo.row_indices) ++csr_row_counter[row_id];

            // prefix scan to get the starting indices of cols in a row.
            std::exclusive_scan(csr_row_counter.begin(), csr_row_counter.end(), retval.row_start_offsets.begin(), 0);

            // reuse the csr_row_counter as offset counter for rows.
            std::ranges::fill(csr_row_counter, 0);

            // transform to CSR.
            for (size_t i = 0; i < coo.values.size(); ++i) {
                const auto row = coo.row_indices[i];
                const auto row_start = retval.row_start_offsets[row];
                const auto offset = row_start + csr_row_counter[row]++;

                retval.col_indices[offset] = coo.col_indices[i];
                retval.values[offset] = coo.values[i];
            }

            return retval;
        }
    };

} // namespace matrix_storage_formats
} // namespace cg
#endif /* APPS_CONJUGATE_GRADIENT_MATRIX_STORAGE_FORMATS */
