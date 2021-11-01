#ifndef APPS_CSR5_BENCH_SPMV_ALGOS
#define APPS_CSR5_BENCH_SPMV_ALGOS

#ifdef CUDA_ENABLED
#include "cuda_algo_facade.h"
#endif

#include "timed_section.h"
#include <anonymouslib_avx2.h>
#include <dim/mat/storage_formats.h>
#include <dim/span.h>
#include <numeric>

namespace cg::spmv_algos {

template<typename ValueType, template<typename> typename StorageTy>
void cpu_sequential(const dim::mat::csr<ValueType, StorageTy>& matrix, dim::span<ValueType> rhs,
                    dim::span<ValueType> output) {
    for (size_t row = 0; row < matrix.dimensions.rows; ++row) {
        double sum{};
        for (auto i = matrix.row_start_offsets[row]; i < matrix.row_start_offsets[row + 1]; ++i) {
            const auto column = matrix.col_indices[i];
            const auto value = matrix.values[i];

            if (row == 494812)
                fmt::print("{}, ", value);
            sum += rhs[column] * value;
        }

        if (row == 494812)
            fmt::print(": {}\n", sum);

        output[row] = sum;
    }
}

template<typename ValueType>
auto create_csr5_handle(dim::mat::csr<ValueType, dim::mat::cache_aligned_vector>& matrix) {
    csr5::avx2::anonymouslibHandle<int, unsigned int, ValueType> handle{matrix.dimensions.rows, matrix.dimensions.cols};

    handle.inputCSR(matrix.values.size(), reinterpret_cast<int*>(matrix.row_start_offsets.data()),
                    reinterpret_cast<int*>(matrix.col_indices.data()), matrix.values.data());
    handle.setSigma(csr5::avx2::ANONYMOUSLIB_CSR5_SIGMA);

    report_timed_section("CSR5 conversion", [&] { handle.asCSR5(); });

    return handle;
}

template<typename ValueType>
void cpu_avx2(csr5::avx2::anonymouslibHandle<int, unsigned int, ValueType>& A, dim::span<ValueType> rhs,
              dim::span<ValueType> output) {
    A.setX(rhs.data());
    A.spmv(1.0, output.data());
}

template<typename ValueType>
void cpu_avx2(dim::mat::csr<ValueType, dim::mat::cache_aligned_vector>& matrix, dim::span<ValueType> rhs,
              dim::span<ValueType> output) {
    csr5::avx2::anonymouslibHandle<int, unsigned int, ValueType> handle{static_cast<int>(matrix.dimensions.rows),
                                                                        static_cast<int>(matrix.dimensions.cols)};

    handle.inputCSR(static_cast<int>(matrix.values.size()), reinterpret_cast<int*>(matrix.row_start_offsets.data()),
                    reinterpret_cast<int*>(matrix.col_indices.data()), matrix.values.data());
    handle.setSigma(csr5::avx2::ANONYMOUSLIB_CSR5_SIGMA);
    handle.asCSR5();

    cpu_avx2(handle, rhs, output);
}

#ifdef CUDA_ENABLED
void cuda_complete_bench(dim::mat::csr<double, dim::mat::cache_aligned_vector>& matrix, dim::span<double> rhs,
                         dim::span<double> output) {
    bench_csr5_cuda(static_cast<int>(matrix.dimensions.rows), static_cast<int>(matrix.dimensions.cols),
                    static_cast<int>(matrix.values.size()), reinterpret_cast<int*>(matrix.row_start_offsets.data()),
                    reinterpret_cast<int*>(matrix.col_indices.data()), matrix.values.data(), rhs.data(), output.data());
}
#endif

} // namespace cg::spmv_algos

#endif /* APPS_CSR5_BENCH_SPMV_ALGOS */
