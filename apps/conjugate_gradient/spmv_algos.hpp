#ifndef APPS_CONJUGATE_GRADIENT_SPMV_ALGOS
#define APPS_CONJUGATE_GRADIENT_SPMV_ALGOS

#include "matrix_storage_formats.h"
#include <bits/ranges_algo.h>
#include <numeric>
#include <ranges>
#include <span>

namespace cg::spmv_algos {

template<typename ValueType, template<typename> typename StorageTy>
void cpu_sequential(const matrix_storage_formats::csr<ValueType, StorageTy>& matrix, std::span<ValueType> rhs,
         std::span<ValueType> output) {
    for (size_t row = 0; row < matrix.dimensions.rows; ++row) {
        double sum{};
        for (auto i = matrix.row_start_offsets[row]; i < matrix.row_start_offsets[row + 1]; ++i) {
            const auto column = matrix.col_indices[i];
            const auto value = matrix.values[i];
            sum += rhs[column] * value;
        }

        output[row] = sum;
    }
}

} // namespace cg::spmv_algos

#endif /* APPS_CONJUGATE_GRADIENT_SPMV_ALGOS */
