#include "dim/bench/stopwatch.h"
#include <dim/bench/timed_section.h>
#include <dim/io/h5.h>
#include <dim/simple_main.h>
#include <dim/vec.h>

#include <structopt/app.hpp>

#include <spdlog/stopwatch.h>

#include <nlohmann/json.hpp>

#include <fmt/chrono.h>
#include <fmt/ranges.h>

#include <omp.h>

#include <algorithm>
#include <execution>
#include <fstream>
#include <numeric>

namespace h5 = dim::io::h5;
namespace fs = std::filesystem;

struct arguments_t
{
    fs::path input_file;
    std::optional<std::string> group_name{"A"};
    std::optional<double> threshold{0.1};
    std::optional<size_t> max_iters{100};
};
STRUCTOPT(arguments_t, input_file);

struct general_stats_t
{
    size_t thread_count{static_cast<size_t>(::omp_get_max_threads())};
};

void to_json(nlohmann::json& j, const general_stats_t& stats) {
    j = nlohmann::json{
      {"thread_count", stats.thread_count},
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
        second alpha{};
        second x{};
        second r{};
        second s{};
        second r_r{};
    } steps;
};

void to_json(nlohmann::json& j, const cg_stats_t& stats) {
    j = nlohmann::json{{"num_iters", stats.num_iters},
                       {"total", stats.total.count()},
                       {"steps",
                        {{"A*s", stats.steps.spmv.count()},
                         {"alpha = (r*r) / s*A*s", stats.steps.alpha.count()},
                         {"x += s * alpha", stats.steps.x.count()},
                         {"r -= temp * alpha", stats.steps.r.count()},
                         {"sk+1 = rk+1 + beta * sk;", stats.steps.s.count()},
                         {"r*r", stats.steps.r_r.count()}}}};
}

using dim::bench::second;
using dim::bench::section;
using dim::bench::stopwatch;

int main_impl(const arguments_t& args) {
    const auto general_stats = general_stats_t{};

    const auto global_sw = stopwatch{};

    auto sw = stopwatch{};
    const auto matrix = h5::load_csr5(args.input_file, *args.group_name);
    const auto io_time = sw.elapsed();

    const auto element_count = matrix.dimensions.rows;

    // r0 = b - A * x0
    // since we're using x0 == {0, ...}, r0 = b, so we just reuse b.
    auto r = dim::vec(element_count, 1.);
    auto s = dim::vec(element_count, 1.);
    auto x = dim::vec(element_count, 0.);

    auto r_r = dot(r, r);
    const auto norm_b = std::sqrt(r_r);

    auto As = dim::vec(element_count, 0.);
    auto calibrator = matrix.allocate_calibrator();

    auto cg_stats = cg_stats_t{};
    const auto cg_sw = stopwatch{};
    for (size_t i = 0; i < *args.max_iters; ++i) {
        spdlog::info("running iteration {}", i);

        // end condition.
        if (std::sqrt(r_r) / norm_b <= *args.threshold)
            break;

        // temp = A*s
        cg_stats.steps.spmv += section([&] { matrix.spmv({.x = s.raw(), .y = As.raw(), .calibrator = calibrator}); });

        // alpha = (r * r)  / (s * temp)
        sw.reset();
        const auto alpha = r_r / dot(s, As);
        cg_stats.steps.alpha += sw.elapsed();

        // x += s * alpha
        cg_stats.steps.x += section([&] { x.axpy(s, alpha); });

        // r -= temp * alpha
        cg_stats.steps.r += section([&] { r.axpy(As, -alpha); });

        sw.reset();
        auto tmp = dot(r, r);
        cg_stats.steps.r_r += sw.elapsed();

        auto beta = tmp / std::exchange(r_r, tmp);

        // sk+1 = rk+1 + beta * sk;
        cg_stats.steps.s += section([&] { s.aypx(r, beta); });
    }
    cg_stats.num_iters = *args.max_iters;
    cg_stats.total = cg_sw.elapsed();
    const auto global_time = global_sw.elapsed();

    std::ofstream{"stats.json"} << std::setw(4)
                                << nlohmann::json({
                                     {"total", global_time.count()},
                                     {"general_info", general_stats},
                                     {"cg", cg_stats},
                                     {"io", io_time.count()},
                                   });

    return 0;
}
DIM_MAIN(arguments_t);