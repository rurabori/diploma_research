#include <benchmark/benchmark.h>

#include <dim/io/h5.h>
#include <dim/io/matrix_market.h>

#include <anonymouslib_avx2.h>

using dim::io::h5::read_matlab_compatible;

static void bm_csr5_orig(benchmark::State& state) {
    auto csr = read_matlab_compatible("./resources/matrices/nlpkkt200/nlpkkt200.csr.h5", "A");

    using csr5_handle_t = anonymouslibHandle<int, unsigned int, double>;

    auto handle = csr5_handle_t{static_cast<int>(csr.dimensions.rows), static_cast<int>(csr.dimensions.cols)};

    // the casts are necessary as CSR5 uses signed types regardless of if signed values are ever valid.
    handle.setSigma(ANONYMOUSLIB_CSR5_SIGMA);
    handle.inputCSR(static_cast<int>(csr.values.size()), reinterpret_cast<int*>(csr.row_start_offsets.data()),
                    reinterpret_cast<int*>(csr.col_indices.data()), csr.values.data());
    handle.asCSR5();

    auto x = dim::mat::cache_aligned_vector<double>(csr.dimensions.cols, 1.25);
    handle.setX(x.data());

    auto Y = dim::mat::cache_aligned_vector<double>(csr.dimensions.cols, 0);

    for (auto _ : state) {
        handle.spmv(0, Y.data());
        benchmark::DoNotOptimize(Y);
    }

    handle.destroy();
}
BENCHMARK(bm_csr5_orig);

static void bm_csr5_dim(benchmark::State& state) {
    auto matrix = dim::io::h5::load_csr5("./resources/matrices/nlpkkt200/nlpkkt200.csr5.h5", "A");

    const auto x = dim::mat::cache_aligned_vector<double>(matrix.dimensions.cols, 1.25);
    auto Y = dim::mat::cache_aligned_vector<double>(matrix.dimensions.cols, 0);
    auto calibrator = matrix.create_calibrator();

    for (auto _ : state) {
        matrix.spmv({x, Y, calibrator});
        benchmark::DoNotOptimize(Y);
    }
}
BENCHMARK(bm_csr5_dim);
