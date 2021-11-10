#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iterator>
#include <random>
#include <stdexcept>
#include <vector>

#include <fmt/chrono.h>
#include <fmt/format.h>

#include <anonymouslib_avx2.h>

#include <magic_enum.hpp>

#include <scn/scn.h>
#include <stx/panic.h>

#include "arguments.h"

#include <dim/io/h5.h>
#include <dim/io/matrix_market.h>
#include <dim/mat/storage_formats/csr.h>
#include <dim/mat/storage_formats/csr5.h>

#include "spmv_algos.hpp"
#include "timed_section.h"

#include "version.h"

using dim::mat::cache_aligned_vector;

auto generate_random_vector(size_t required_size) {
    cache_aligned_vector<double> x(required_size, 0.);

    std::mt19937_64 prng{std::random_device{}()};
    std::uniform_real_distribution<double> dist{0., 1.};
    std::generate(x.begin(), x.end(), [&] { return dist(prng); });

    return x;
}

int main(int argc, char* argv[]) {
    auto app = structopt::app(brr::app_info.full_name, brr::app_info.version);
    auto arguments = app.parse<arguments_t>(argc, argv);

    namespace h5 = dim::io::h5;

    auto in = h5::file_t::open(arguments.input_file, H5F_ACC_RDONLY);
    auto group = in.open_group("A");
    auto matrix = dim::io::h5::read_matlab_compatible(group.get_id());

    auto consumed_memory
      = matrix.col_indices.size() * sizeof(typename decltype(matrix.col_indices)::value_type)
        + matrix.row_start_offsets.size() * sizeof(typename decltype(matrix.row_start_offsets)::value_type)
        + matrix.values.size() * sizeof(typename decltype(matrix.values)::value_type);

    fmt::print("consumed_memory={}\n", consumed_memory);

    auto dimensions = matrix.dimensions;

    auto x = std::vector<double>(dimensions.cols, 1.);

    auto Y = cache_aligned_vector<double>(matrix.dimensions.rows, 0.);
    switch (*arguments.algorithm) {
        case arguments_t::algorithm_t::cpu_sequential: {
            report_timed_section("SpMV", [&] { cg::spmv_algos::cpu_sequential(matrix, dim::span{x}, dim::span{Y}); });
            break;
        }
        case arguments_t::algorithm_t::cpu_avx2: {
            const auto csr5 = dim::mat::csr5<double>::from_csr(matrix);

            report_timed_section("SpMV", [&] { csr5.spmv(dim::span{x}, dim::span{Y}); });
            break;
        }
    }

    auto file = h5::file_t::create("o.h5", H5F_ACC_TRUNC);
    auto dataset = file.create_dataset("Y", H5T_IEEE_F64LE, h5::dataspace_t::create(hsize_t{Y.size()}));
    dataset.write(Y.data(), H5T_NATIVE_DOUBLE);

    if (*arguments.debug) {
        constexpr auto comparator = [](auto l, auto r) {
            return std::abs(l - r) <= std::max(std::abs(r) * std::numeric_limits<double>::epsilon() * 1E12,
                                               std::numeric_limits<double>::epsilon() * 1E12);
        };

        auto Y_ref = cache_aligned_vector<double>(matrix.dimensions.rows, 0.);
        cg::spmv_algos::cpu_sequential(matrix, dim::span{x}, dim::span{Y_ref});

        // const auto correct = std::equal(Y.begin(), Y.end(), Y_ref.begin(),
        //                                 [](auto l, auto r) { return std::abs(l - r) <= (0.01 * std::abs(r));
        //                                 });
        // fmt::print(stderr, "SpMV correct: {}\n", correct);

        for (size_t i = 0; i < Y_ref.size(); ++i) {
            const auto l = Y_ref[i];
            const auto r = Y[i];

            if (!comparator(l, r))
                fmt::print("idx: {} got: {}, expected {}\n", i, r, l);
        }
    }

    return 0;
}
