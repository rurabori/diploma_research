#include "version.h"

#include <dim/csr5_mpi/csr5_mpi.h>
#include <dim/io/h5.h>
#include <dim/mpi/mpi.h>
#include <dim/simple_main.h>

#include <structopt/app.hpp>

#include <fmt/chrono.h>
#include <fmt/ranges.h>

#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <execution>
#include <filesystem>
#include <limits>
#include <numeric>

namespace fs = std::filesystem;

struct arguments_t
{
    fs::path input_file;
    std::optional<std::string> group_name{"A"};
    std::optional<double> threshold{0.1};
    std::optional<size_t> max_iters{100};
};
STRUCTOPT(arguments_t, input_file);

auto mpi_reduce(std::span<const double> a, std::span<const double> b) -> double {
    // we now have a * b partial result.
    const auto partial = std::transform_reduce(std::execution::par_unseq, a.begin(), a.end(), b.begin(), 0.);

    auto result = double{};
    ::MPI_Allreduce(&partial, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return result;
}

auto mpi_reduce(std::span<const double> a) -> double { return mpi_reduce(a, a); }

auto mpi_magnitude(std::span<const double> b) -> double { return std::sqrt(mpi_reduce(b, b)); }

constexpr auto op_equals = [](auto&& lhs, auto&& rhs, auto&& alpha, auto&& op) {
    std::transform(std::execution::par_unseq, std::begin(lhs), std::end(lhs), std::begin(rhs), std::begin(lhs),
                   [alpha, op](auto&& lhs, auto&& rhs) { return op(lhs, alpha * rhs); });
};

auto main_impl(const arguments_t& args) -> int {
    using dim::csr5_mpi::csr5_partial;
    constexpr auto csr5_strat = csr5_partial::matrix_type::spmv_strategy::partial;

    spdlog::stopwatch sw;

    const auto mpi_mat = csr5_partial::load(args.input_file, *args.group_name);
    spdlog::info("loading matrix took {}", sw.elapsed());

    const auto [first_row, last_row] = mpi_mat.output_range();
    auto [edge_sync, result_sync] = mpi_mat.make_sync();

    // load the relevant part of b.
    auto b = dim::mat::cache_aligned_vector<double>(mpi_mat.matrix.dimensions.cols, 1.);

    const auto element_count = last_row - first_row + 1;

    // r0 = b - A * x0
    // since we're using x0 == {0, ...}, r0 = b, so we just reuse b.
    const auto all_b = std::span{b}.subspan(first_row, element_count);

    // conditionally skip first element.
    const auto owned_b = all_b.subspan(mpi_mat.matrix.first_tile_uncapped());

    // norm_b = b.magnitude()
    const auto norm_b = mpi_magnitude(owned_b);

    auto temp = dim::mat::cache_aligned_vector<double>(element_count, 0.);
    const auto owned_temp = std::span{temp}.subspan(mpi_mat.matrix.first_tile_uncapped());

    auto calibrator = mpi_mat.matrix.allocate_calibrator();
    auto x_partial = dim::mat::cache_aligned_vector<double>(element_count, 0.);

    auto err = std::numeric_limits<double>::max();
    for (size_t i = 0; i < *args.max_iters && err >= *args.threshold; ++i) {
        sw.reset();
        // temp = A*r
        mpi_mat.matrix.spmv<csr5_strat>({.x = b, .y = temp, .calibrator = calibrator});
        spdlog::info("spmv took {}", sw.elapsed());

        sw.reset();
        edge_sync.sync(temp);
        spdlog::info("edge sync took {}", sw.elapsed());

        // alpha = (r * r)  / (r * temp)
        const auto alpha = mpi_reduce(owned_b) / mpi_reduce(owned_b, owned_temp);

        // x += r * alpha
        op_equals(x_partial, owned_b, alpha, std::plus<>{});

        // r -= temp * alpha
        op_equals(owned_b, owned_temp, alpha, std::minus<>{});

        err = mpi_magnitude(owned_b) / norm_b;
        spdlog::info("err is {}", err);

        sw.reset();
        result_sync.sync(b, owned_b);
        spdlog::info("syncing residual took {}", sw.elapsed());
    }

    return 0;
}

int main(int argc, char* argv[]) try {
    dim::mpi::ctx ctx{argc, argv};
    auto app = structopt::app(brr::app_info.full_name, brr::app_info.version);
    spdlog::set_default_logger(
      spdlog::stdout_logger_mt(fmt::format("{} (n{:02})", brr::app_info.full_name, dim::mpi::rank())));

    return main_impl(app.parse<arguments_t>(argc, argv));
} catch (const structopt::exception& e) {
    spdlog::critical("{}", e.what());
    fmt::print(stderr, "{}", e.help());
    return 1;
} catch (const std::exception& e) {
    spdlog::critical("{}", e.what());
    return 2;
}
