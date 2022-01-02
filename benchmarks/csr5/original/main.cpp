#include "dim/mat/storage_formats/base.h"
#include <dim/simple_main.h>

#include <dim/io/h5.h>

#include <filesystem>

#include <anonymouslib_avx2.h>
#include <spdlog/stopwatch.h>

struct arguments_t
{
    std::filesystem::path input;
    std::optional<std::string> group{"/A"};
};
STRUCTOPT(arguments_t, input, group);

auto main_impl(const arguments_t& args) {
    auto csr = dim::io::h5::read_matlab_compatible(args.input, *args.group);

    using csr5_handle_t = anonymouslibHandle<int, unsigned int, double>;

    auto handle = csr5_handle_t{static_cast<int>(csr.dimensions.rows), static_cast<int>(csr.dimensions.cols)};

    spdlog::stopwatch sw;
    // the casts are necessary as CSR5 uses signed types regardless of if signed values are ever valid.
    handle.setSigma(ANONYMOUSLIB_CSR5_SIGMA);
    handle.inputCSR(static_cast<int>(csr.values.size()), reinterpret_cast<int*>(csr.row_start_offsets.data()),
                    reinterpret_cast<int*>(csr.col_indices.data()), csr.values.data());
    handle.asCSR5();
    spdlog::info("conversion to CSR5 took {}s", sw);

    auto x = dim::mat::cache_aligned_vector<double>(csr.dimensions.cols, 1.);
    handle.setX(x.data());

    auto Y = dim::mat::cache_aligned_vector<double>(csr.dimensions.cols, 0);

    sw.reset();
    handle.spmv(0., Y.data());
    spdlog::info("spmv took {}s", sw);

    handle.destroy();

    return 0;
}
DIM_MAIN(arguments_t);
