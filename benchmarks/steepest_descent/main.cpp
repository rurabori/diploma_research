#include <dim/io/h5.h>
#include <dim/simple_main.h>

#include <structopt/app.hpp>

#include <spdlog/stopwatch.h>

#include <fmt/chrono.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <execution>
#include <numeric>

namespace h5 = dim::io::h5;
namespace fs = std::filesystem;

struct arguments_t
{
    fs::path input_file;
    std::optional<std::string> group_name{"A"};
    std::optional<double> threshold{0.1};
};
STRUCTOPT(arguments_t, input_file);

auto vec_multiply(std::span<const double> a, std::span<const double> b) -> double {
    return std::transform_reduce(std::execution::par_unseq, a.begin(), a.end(), b.begin(), 0.);
}

auto magnitude(std::span<const double> a) -> double { return std::sqrt(vec_multiply(a, a)); }

constexpr auto op_equals = [](auto&& lhs, auto&& rhs, auto&& alpha, auto&& op) {
    std::transform(std::execution::par_unseq, std::begin(lhs), std::end(lhs), std::begin(rhs), std::begin(lhs),
                   [alpha, op](auto&& lhs, auto&& rhs) { return op(lhs, alpha * rhs); });
};

int main_impl(const arguments_t& args) {
    spdlog::stopwatch sw;
    const auto matrix = h5::load_csr5(args.input_file, *args.group_name);
    spdlog::info("loading matrix took {}", sw.elapsed());

    using csr5_t = decltype(matrix);

    const auto element_count = matrix.dimensions.rows;

    // r0 = b - A * x0
    // since we're using x0 == {0, ...}, r0 = b, so we just reuse b.
    auto b = dim::mat::cache_aligned_vector<double>(element_count, 1.);
    const auto norm_b = magnitude(b);
    spdlog::info("norm b {}", norm_b);

    auto temp = dim::mat::cache_aligned_vector<double>(element_count, 0.);
    auto calibrator = matrix.allocate_calibrator();

    auto x = dim::mat::cache_aligned_vector<double>(matrix.dimensions.cols, 0.);

    auto err = double{};
    do {
        sw.reset();
        // temp = A*r
        matrix.spmv<csr5_t::spmv_strategy::absolute>({.x = b, .y = temp, .calibrator = calibrator});
        spdlog::info("spmv took {}", sw.elapsed());

        // alpha = (r * r)  / (r * temp)
        const auto alpha = vec_multiply(b, b) / vec_multiply(b, temp);

        spdlog::info("computed alpha {}", alpha);

        // x += r * alpha
        op_equals(x, b, alpha, std::plus<>{});

        // r -= temp * alpha
        op_equals(b, temp, alpha, std::minus<>{});
        spdlog::info("b: {}", std::span{b}.subspan(0, 5));

        const auto mag = magnitude(b);
        spdlog::info("mag is {}", mag);

        err = mag / norm_b;
        spdlog::info("err is {}", err);
    } while (err >= *args.threshold);

    return 0;
}
DIM_MAIN(arguments_t);