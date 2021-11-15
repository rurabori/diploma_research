#include <benchmark/benchmark.h>

#include <dim/io/h5.h>
#include <dim/io/matrix_market.h>

using dim::io::h5::read_matlab_compatible;
using dim::io::matrix_market::load_as_csr;

static void bm_io_matrix_market(benchmark::State& state) {
    for (auto _ : state) {
        auto matrix = load_as_csr<double>("./resources/matrices/kmer_V1r/kmer_V1r.mtx");
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(bm_io_matrix_market);

static void bm_io_h5(benchmark::State& state) {
    for (auto _ : state) {
        auto matrix = read_matlab_compatible("./resources/matrices/e40r5000.csr.h5", "A");
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(bm_io_h5);
