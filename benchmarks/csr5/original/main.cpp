#include <dim/bench/timed_section.h>
#include <dim/io/h5.h>
#include <dim/simple_main.h>

#include <anonymouslib_avx2.h>

#include <spdlog/spdlog.h>

#include <fmt/chrono.h>

#include <filesystem>

struct arguments_t
{
    std::filesystem::path input;
    std::optional<std::string> group{"/A"};
    std::optional<size_t> num_runs{100};
};
STRUCTOPT(arguments_t, input, group, num_runs);

auto main_impl(const arguments_t& args) {
    auto csr = dim::io::h5::read_matlab_compatible(args.input, *args.group);

    using csr5_handle_t = anonymouslibHandle<int, unsigned int, double>;

    auto handle = csr5_handle_t{static_cast<int>(csr.dimensions.rows), static_cast<int>(csr.dimensions.cols)};

    dim::bench::stopwatch sw;
    // the casts are necessary as CSR5 uses signed types regardless of if signed values are ever valid.
    handle.setSigma(ANONYMOUSLIB_CSR5_SIGMA);
    handle.inputCSR(static_cast<int>(csr.values.size()), reinterpret_cast<int*>(csr.row_start_offsets.data()),
                    reinterpret_cast<int*>(csr.col_indices.data()), csr.values.data());
    handle.asCSR5();
    spdlog::info("conversion to CSR5 took {}s", sw.elapsed());

    auto x = dim::mat::cache_aligned_vector<double>(csr.dimensions.cols, 1.);
    handle.setX(x.data());

    auto Y = dim::mat::cache_aligned_vector<double>(csr.dimensions.cols, 0);

    for (size_t i = 0; i < 5; ++i)
        handle.spmv(0., Y.data());

    auto total_time = dim::bench::second{0};

    for (size_t i = 0; i < *args.num_runs; ++i) {
        spdlog::info("running iteration {}", i);
        const auto it_time = dim::bench::section([&] { handle.spmv(0., Y.data()); });
        total_time += it_time;
        spdlog::info("iteration {} took {}", i, it_time);
    }
    spdlog::info("spmv took {} on average", total_time / *args.num_runs);

    handle.destroy();

    return 0;
}
DIM_MAIN(arguments_t);
