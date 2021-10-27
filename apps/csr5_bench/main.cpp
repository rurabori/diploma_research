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

#include <tclap/Arg.h>
#include <tclap/CmdLine.h>
#include <tclap/SwitchArg.h>
#include <tclap/ValueArg.h>

#include <magic_enum.hpp>

#include <scn/scn.h>
#include <stx/panic.h>

#include "dim/io/h5.h"
#include "dim/mat/storage_formats/csr.h"
#include "dim/mat/storage_formats/csr5.h"
#include "spmv_algos.hpp"
#include "timed_section.h"

#include "version.h"

#include <dim/io/matrix_market.h>

using dim::mat::cache_aligned_vector;

auto generate_random_vector(size_t required_size) {
    cache_aligned_vector<double> x(required_size, 0.);

    std::mt19937_64 prng{std::random_device{}()};
    std::uniform_real_distribution<double> dist{0., 1.};
    std::generate(x.begin(), x.end(), [&] { return dist(prng); });

    return x;
}

struct arguments
{
    enum class algorithm_t
    {
        cpu_sequential,
        cpu_avx2,
#ifdef CUDA_ENABLED
        cuda,
#endif
    };

    std::filesystem::path matrix_file{};
    bool debug{};
    algorithm_t algorithm;

    static algorithm_t get_algoritm(std::string_view string_representation) {
        if (auto algo = magic_enum::enum_cast<algorithm_t>(string_representation); algo)
            return *algo;

        return algorithm_t::cpu_sequential;
    }

    static arguments from_main(int argc, const char* argv[]) {
        TCLAP::CmdLine commandline{"Conjugate Gradient.", ' ', csr5_bench_VER};

        TCLAP::ValueArg<std::string> matrix_name_arg{"m", "matrix-path", "Path to matrix", true, "", "string"};
        commandline.add(matrix_name_arg);

        TCLAP::SwitchArg debug_arg{"d", "debug", "Enable debug output", false};
        commandline.add(debug_arg);

        TCLAP::ValueArg<std::string> algorithm_arg{"a",   "algorithm",      "Algorithm to use for SpMV",
                                                   false, "cpu_sequential", "one of [cpu_sequential, cpu_avx2, cuda]"};
        commandline.add(algorithm_arg);

        commandline.parse(argc, argv);
        return arguments{.matrix_file = matrix_name_arg.getValue(),
                         .debug = debug_arg.getValue(),
                         .algorithm = get_algoritm(algorithm_arg.getValue())};
    }
};

int main(int argc, const char* argv[]) {
    auto arguments = arguments::from_main(argc, argv);
    namespace h5 = dim::io::h5;

    auto in = h5::file_t{::H5Fopen(arguments.matrix_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT)};
    auto group = h5::group_t{::H5Gopen(in.get(), "A", H5P_DEFAULT)};
    auto matrix = dim::io::h5::read_matlab_compatible(group.get());

    auto consumed_memory
      = matrix.col_indices.size() * sizeof(typename decltype(matrix.col_indices)::value_type)
        + matrix.row_start_offsets.size() * sizeof(typename decltype(matrix.row_start_offsets)::value_type)
        + matrix.values.size() * sizeof(typename decltype(matrix.values)::value_type);

    fmt::print("consumed_memory={}\n", consumed_memory);

    auto dimensions = matrix.dimensions;

    auto x = std::vector<double>(dimensions.cols, 100.);

    auto Y = cache_aligned_vector<double>(matrix.dimensions.rows, 0.);
    switch (arguments.algorithm) {
        case arguments::algorithm_t::cpu_sequential: {
            report_timed_section("SpMV", [&] { cg::spmv_algos::cpu_sequential(matrix, dim::span{x}, dim::span{Y}); });
            break;
        }
        case arguments::algorithm_t::cpu_avx2: {
            const auto csr5 = dim::mat::csr5<double>::from_csr(matrix);

            report_timed_section("SpMV", [&] { csr5.spmv(dim::span{x}, dim::span{Y}); });
            break;
        }
#ifdef CUDA_ENABLED
        case arguments::algorithm_t::cuda: {
            cg::spmv_algos::cuda_complete_bench(matrix, dim::span{x}, dim::span{Y});
            break;
        }
#endif
    }

    if (arguments.debug) {
        constexpr auto comparator
          = [](auto l, auto r) { return std::abs(l - r) <= std::numeric_limits<double>::epsilon() * 1E5; };

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
