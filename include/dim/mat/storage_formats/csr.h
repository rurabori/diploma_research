#ifndef INCLUDE_DIM_MAT_STORAGE_FORMATS_CSR
#define INCLUDE_DIM_MAT_STORAGE_FORMATS_CSR

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <omp.h>
#include <span>
#include <vector>

#include <dim/keyval_iterator.h>
#include <dim/mat/storage_formats/base.h>
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

    csr(dimensions_t dimensions_, size_t non_zero_count_)
      : dimensions{dimensions_},
        values(non_zero_count_),
        row_start_offsets(dimensions_.rows + 1),
        col_indices(non_zero_count_) {}

    csr(dimensions_t dimensions_, values_t values_, indices_t row_start_offsets_, indices_t col_indices_)
      : dimensions{dimensions_},
        values{std::move(values_)},
        row_start_offsets{std::move(row_start_offsets_)},
        col_indices{std::move(col_indices_)} {}

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

#pragma omp parallel for
        for (size_t idx = 1; idx < retval.row_start_offsets.size(); ++idx) {
            const auto row_start = retval.row_start_offsets[idx - 1];
            const auto row_end = retval.row_start_offsets[idx];

            keyval_sort(std::span{retval.col_indices}.subspan(row_start, row_end - row_start),
                        std::span{retval.values}.subspan(row_start, row_end - row_start));
        }

        return retval;
    }
};

template<typename CsrType = csr<double>>
struct csr_partial_t
{
    //! @brief dimensions of the full matrix this one is a chunk off.
    dimensions_t global_dimensions{};
    //! @brief index of first row which is contained in this chunk.
    size_t first_row{};
    //! @brief chunk of full matrix.
    CsrType matrix_chunk;
};

} // namespace dim::mat

#endif /* INCLUDE_DIM_MAT_STORAGE_FORMATS_CSR */
