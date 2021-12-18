#include <dim/bench/stopwatch.h>
#include <dim/bench/timed_section.h>
#include <dim/csr5_mpi/csr5_mpi.h>
#include <dim/mpi/mpi.h>

#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <fmt/chrono.h>

#include <structopt/app.hpp>

#include <nlohmann/json.hpp>

#include <chrono>
#include <execution>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#include "version.h"

struct arguments_t
{
    std::filesystem::path input_file;
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

constexpr auto a_inner = [](auto&& y, auto&& x, auto&& op) {
    std::transform(std::execution::par_unseq, std::begin(y), std::end(y), std::begin(x), std::begin(y), op);
};

constexpr auto axpy = [](auto&& y, auto&& x, auto&& alpha) {
    a_inner(y, x, [alpha](auto&& lhs_elem, auto&& rhs_elem) { return lhs_elem + alpha * rhs_elem; });
};

constexpr auto aypx = [](auto&& y, auto&& x, auto&& alpha) {
    a_inner(y, x, [alpha](auto&& lhs_elem, auto&& rhs_elem) { return rhs_elem + alpha * lhs_elem; });
};

using stopwatch = dim::bench::stopwatch;
using dim::bench::section;

auto main_impl(const arguments_t& args) -> int {
    using dim::csr5_mpi::csr5_partial;
    constexpr auto csr5_strat = csr5_partial::matrix_type::spmv_strategy::partial;

    auto global_sw = stopwatch{};

    auto sw = stopwatch{};
    const auto mpi_mat = csr5_partial::load(args.input_file, *args.group_name);
    const auto matrix_load_time = sw.elapsed();

    sw.reset();
    const auto output_range = mpi_mat.output_range();
    const auto sync_first_row = mpi_mat.matrix.first_tile_uncapped();
    auto sync = mpi_mat.make_sync();
    const auto sync_time = sw.elapsed();

    const auto element_count = output_range.count();

    // r0 = b - A*x0;  x0 = {0...} => r0 = b
    auto r = dim::mat::cache_aligned_vector<double>(mpi_mat.matrix.dimensions.cols, 1.);
    const auto owned_r = std::span{r}.subspan(output_range.first_row, element_count).subspan(sync_first_row);
    // s0 = r0
    auto s = dim::mat::cache_aligned_vector<double>(mpi_mat.matrix.dimensions.cols, 1.);
    const auto owned_s = std::span{s}.subspan(output_range.first_row, element_count).subspan(sync_first_row);

    // norm_b = b.magnitude()
    const auto norm_b = mpi_magnitude(owned_r);

    // the output vector for spmv.
    auto temp = dim::mat::cache_aligned_vector<double>(element_count, 0.);
    // and a view into it conditionally skipping first element.
    const auto owned_temp = std::span{temp}.subspan(sync_first_row);

    auto calibrator = mpi_mat.matrix.allocate_calibrator();

    auto x_partial = dim::mat::cache_aligned_vector<double>(owned_r.size(), 0.);

    auto r_r = mpi_reduce(owned_r);

    const auto cg_sw = stopwatch{};
    auto spmv_time = dim::bench::second{};
    auto edge_sync_time = dim::bench::second{};
    auto alpha_time = dim::bench::second{};
    auto part_x_time = dim::bench::second{};
    auto part_r_time = dim::bench::second{};
    auto part_s_time = dim::bench::second{};
    auto s_dist_time = dim::bench::second{};
    auto r_r_time = dim::bench::second{};
    for (size_t i = 0; i < *args.max_iters; ++i) {
        // end condition.
        if (std::sqrt(r_r) / norm_b <= *args.threshold)
            break;

        // temp = A*s
        spmv_time += section([&] { mpi_mat.matrix.spmv<csr5_strat>({.x = s, .y = temp, .calibrator = calibrator}); });
        edge_sync_time += section([&] { sync.edge_sync.sync(temp); });

        // alpha = (r * r)  / (s * temp)
        sw.reset();
        const auto alpha = r_r / mpi_reduce(owned_s, owned_temp);
        alpha_time += sw.elapsed();

#pragma omp parallel
        {
#pragma omp single
            {
#pragma omp task
                {
                    // x += s * alpha
                    part_x_time += section([&] { axpy(x_partial, owned_s, alpha); });
                }

#pragma omp task
                {
                    // r -= temp * alpha
                    part_r_time += section([&] { axpy(owned_r, owned_temp, -alpha); });
                }

#pragma omp taskwait
            }
        }

        sw.reset();
        auto tmp = mpi_reduce(owned_r);
        r_r_time += sw.elapsed();

        auto beta = tmp / std::exchange(r_r, tmp);

        // sk+1 = rk+1 + beta * sk;
        part_s_time += section([&] { aypx(owned_s, owned_r, beta); });

        s_dist_time += section([&] { sync.result_sync.sync(s, owned_s); });
    }
    const auto cg_time = cg_sw.elapsed();
    const auto global_time = global_sw.elapsed();

    std::ofstream{fmt::format("stats_{:02}.json", dim::mpi::rank())} << nlohmann::json{
      {"total", global_time.count()},    {"cg", cg_time.count()},         {"io", matrix_load_time.count()},
      {"sync_creat", sync_time.count()}, {"spmv", spmv_time.count()},     {"edge_sync", edge_sync_time.count()},
      {"alpha", alpha_time.count()},     {"part_x", part_x_time.count()}, {"part_r", part_r_time.count()},
      {"part_s", part_s_time.count()},   {"s_dist", s_dist_time.count()}, {"r_r", r_r_time.count()}};

    return 0;
}
int main(int argc, char* argv[]) try {
    dim::mpi::ctx ctx{argc, argv};
    auto app = structopt::app(brr::app_info.full_name, brr::app_info.version);
    spdlog::set_default_logger(
      spdlog::stderr_logger_mt(fmt::format("{} (n{:02})", brr::app_info.full_name, dim::mpi::rank())));

    return main_impl(app.parse<arguments_t>(argc, argv));
} catch (const structopt::exception& e) {
    spdlog::critical("{}", e.what());
    fmt::print(stderr, "{}", e.help());
    return 1;
} catch (const std::exception& e) {
    spdlog::critical("{}", e.what());
    return 2;
}