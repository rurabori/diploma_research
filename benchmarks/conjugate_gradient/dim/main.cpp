#include <dim/bench/stopwatch.h>
#include <dim/bench/timed_section.h>
#include <dim/mpi/csr5.h>
#include <dim/mpi/mpi.h>
#include <dim/vec.h>

#include <omp.h>
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

using stopwatch = dim::bench::stopwatch;
using dim::bench::section;

struct general_stats_t
{
    size_t local_rows;
    size_t local_elements;
    size_t node{dim::mpi::rank()};
    size_t node_count{dim::mpi::size()};
    size_t thread_count{static_cast<size_t>(::omp_get_max_threads())};
};

void to_json(nlohmann::json& j, const general_stats_t& stats) {
    j = nlohmann::json{
      {"local_rows", stats.local_rows}, {"local_elements", stats.local_elements}, {"node", stats.node},
      {"node_count", stats.node_count}, {"thread_count", stats.thread_count},
    };
}

struct cg_stats_t
{
    using second = dim::bench::second;

    size_t num_iters{};
    second total{};

    struct steps_t
    {
        second spmv{};
        second edge_sync{};
        second alpha{};
        second part_x{};
        second part_r{};
        second part_s{};
        second s_dist{};
        second r_r{};
    } steps;
};

void to_json(nlohmann::json& j, const cg_stats_t& stats) {
    j = nlohmann::json{{"num_iters", stats.num_iters},
                       {"total", stats.total.count()},
                       {"steps",
                        {{"A*s", stats.steps.spmv.count()},
                         {"edge sync", stats.steps.edge_sync.count()},
                         {"alpha = (r*r) / s*A*s", stats.steps.alpha.count()},
                         {"x += s * alpha", stats.steps.part_x.count()},
                         {"r -= temp * alpha", stats.steps.part_r.count()},
                         {"sk+1 = rk+1 + beta * sk;", stats.steps.part_s.count()},
                         {"distributing s", stats.steps.s_dist.count()},
                         {"r*r", stats.steps.r_r.count()}}}};
}

auto main_impl(const arguments_t& args) -> int {
    using dim::mpi::csr5::csr5_partial;
    constexpr auto csr5_strat = csr5_partial::matrix_type::spmv_strategy::partial;

    auto general_stats = general_stats_t{};

    auto global_sw = stopwatch{};

    spdlog::info("starting to load matrix");
    auto sw = stopwatch{};
    const auto mpi_mat = csr5_partial::load(args.input_file, *args.group_name);
    const auto matrix_load_time = sw.elapsed();
    spdlog::info("loaded matrix in {}", matrix_load_time);

    spdlog::info("creating synchronization");
    sw.reset();
    const auto output_range = mpi_mat.output_range();
    const auto sync_first_row = mpi_mat.matrix.first_tile_uncapped();
    auto sync = mpi_mat.make_sync();
    const auto sync_time = sw.elapsed();
    spdlog::info("creating synchronization took {}", sync_time);

    const auto row_count = output_range.count();

    general_stats.local_rows = row_count;
    general_stats.local_elements = mpi_mat.matrix.vals.size();

    // r0 = b - A*x0;  x0 = {0...} => r0 = b
    auto r = dim::vec<double>(row_count - sync_first_row, 1.);

    // s0 = r0
    auto s = dim::vec<double>(mpi_mat.matrix.dimensions.cols, 1.);
    auto s_out = s.subview(output_range.first_row, row_count);
    auto s_own = s_out.subview(sync_first_row);

    // norm_b = b.magnitude()
    const auto norm_b = mpi_magnitude(r.raw());

    // the output vector for spmv.
    auto temp = dim::vec<double>(row_count);
    // and a view into it conditionally skipping first element.
    auto temp_own = temp.subview(sync_first_row);

    auto calibrator = mpi_mat.matrix.allocate_calibrator();
    auto x_partial = dim::vec<double>(r.size());

    auto r_r = mpi_reduce(r.raw());

    auto cg_stats = cg_stats_t{};
    const auto cg_sw = stopwatch{};
    for (size_t i = 0; i < *args.max_iters; ++i) {
        spdlog::info("running iteration {}", i);

        // end condition.
        if (std::sqrt(r_r) / norm_b <= *args.threshold)
            break;

        // temp = A*s
        cg_stats.steps.spmv += section(
          [&] { mpi_mat.matrix.spmv<csr5_strat>({.x = s.raw(), .y = temp.raw(), .calibrator = calibrator}); });

        sw.reset();
        // fire edge temp sync request.
        auto sync_request = sync.edge_sync.sync(temp.raw());
        cg_stats.steps.edge_sync += sw.elapsed();

        // alpha = (r * r)  / (s * temp)
        sw.reset();
        const auto alpha = r_r / mpi_reduce(s_out.raw(), temp.raw());
        cg_stats.steps.alpha += sw.elapsed();

        // x += s * alpha
        cg_stats.steps.part_x += section([&] { x_partial.axpy(s_own, alpha); });

        // must be done before this.
        cg_stats.steps.edge_sync += section([&] { sync_request.await(temp.raw()); });

        // r -= temp * alpha
        cg_stats.steps.part_r += section([&] { r.axpy(temp_own, -alpha); });

        sw.reset();
        auto tmp = mpi_reduce(r.raw());
        cg_stats.steps.r_r += sw.elapsed();

        auto beta = tmp / std::exchange(r_r, tmp);

        // sk+1 = rk+1 + beta * sk;
        cg_stats.steps.part_s += section([&] { s_own.aypx(r, beta); });
        cg_stats.steps.s_dist += section([&] { sync.result_sync.sync(s.raw()); });
    }
    cg_stats.total = cg_sw.elapsed();
    cg_stats.num_iters = *args.max_iters;
    const auto global_time = global_sw.elapsed();

    std::ofstream{fmt::format("stats_{:02}.json", dim::mpi::rank())} << std::setw(4)
                                                                     << nlohmann::json({
                                                                          {"total", global_time.count()},
                                                                          {"general_info", general_stats},
                                                                          {"cg", cg_stats},
                                                                          {"io", matrix_load_time.count()},
                                                                          {"sync_creat", sync_time.count()},
                                                                        });

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