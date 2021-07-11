#ifndef APPS_CONJUGATE_GRADIENT_SPMV_ALGOS
#define APPS_CONJUGATE_GRADIENT_SPMV_ALGOS

#ifdef CUDA_ENABLED
#include "cuda_algo_facade.h"
#endif

#include "matrix_storage_formats.h"
#include "timed_section.h"
#include <anonymouslib_avx2.h>
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

template<typename ValueType>
auto create_csr5_handle(matrix_storage_formats::csr<ValueType, matrix_storage_formats::cache_aligned_vector>& matrix) {
    csr5::avx2::anonymouslibHandle<int, unsigned int, ValueType> handle{matrix.dimensions.rows, matrix.dimensions.cols};

    handle.inputCSR(matrix.values.size(), reinterpret_cast<int*>(matrix.row_start_offsets.data()),
                    reinterpret_cast<int*>(matrix.col_indices.data()), matrix.values.data());
    handle.setSigma(csr5::avx2::ANONYMOUSLIB_CSR5_SIGMA);

    report_timed_section("CSR5 conversion", [&] { handle.asCSR5(); });

    return handle;
}

template<typename ValueType>
void cpu_avx2(csr5::avx2::anonymouslibHandle<int, unsigned int, ValueType>& A, std::span<ValueType> rhs,
              std::span<ValueType> output) {
    A.setX(rhs.data());
    A.spmv(1.0, output.data());
}

template<typename ValueType>
void cpu_avx2(matrix_storage_formats::csr<ValueType, matrix_storage_formats::cache_aligned_vector>& matrix,
              std::span<ValueType> rhs, std::span<ValueType> output) {
    csr5::avx2::anonymouslibHandle<int, unsigned int, ValueType> handle{static_cast<int>(matrix.dimensions.rows),
                                                                        static_cast<int>(matrix.dimensions.cols)};

    handle.inputCSR(static_cast<int>(matrix.values.size()), reinterpret_cast<int*>(matrix.row_start_offsets.data()),
                    reinterpret_cast<int*>(matrix.col_indices.data()), matrix.values.data());
    handle.setSigma(csr5::avx2::ANONYMOUSLIB_CSR5_SIGMA);
    handle.asCSR5();

    cpu_avx2(handle, rhs, output);
}

#ifdef CUDA_ENABLED
void cuda_complete_bench(matrix_storage_formats::csr<double, matrix_storage_formats::cache_aligned_vector>& matrix,
                         std::span<double> rhs, std::span<double> output) {
    bench_csr5_cuda(static_cast<int>(matrix.dimensions.rows), static_cast<int>(matrix.dimensions.cols),
                    static_cast<int>(matrix.values.size()), reinterpret_cast<int*>(matrix.row_start_offsets.data()),
                    reinterpret_cast<int*>(matrix.col_indices.data()), matrix.values.data(), rhs.data(), output.data());
}
#endif

} // namespace cg::spmv_algos

#endif /* APPS_CONJUGATE_GRADIENT_SPMV_ALGOS */
