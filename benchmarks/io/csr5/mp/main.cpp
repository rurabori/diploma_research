#include <dim/bench/timed_section.h>
#include <dim/csr5_mpi/csr5_mpi.h>
#include <dim/mpi/mpi.h>

#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <fmt/chrono.h>

#include <structopt/app.hpp>

#include <nlohmann/json.hpp>

#include <omp.h>

#include <fstream>

#include "version.h"

struct arguments_t
{
    std::filesystem::path input_file;
    std::optional<size_t> num_loads{100};
    std::optional<std::string> group{"A"};
};
STRUCTOPT(arguments_t, input_file, num_loads, group);

using dim::csr5_mpi::csr5_partial;
using stopwatch = dim::bench::stopwatch;
using dim::bench::section;

auto main_impl(const arguments_t& args) -> int {
    auto total_time = dim::bench::second{};
    for (size_t i = 0; i < *args.num_loads; ++i)
        total_time += section([&] { const auto mpi_mat = csr5_partial::load(args.input_file, *args.group); });

    if (dim::mpi::rank() != 0)
        return 0;

    std::ofstream{"stats.json"} << std::setw(4)
                                << nlohmann::json{{"matrix", args.input_file.filename().string()},
                                                  {"process_count", dim::mpi::size()},
                                                  {"total", total_time.count()},
                                                  {"avg", total_time.count() / *args.num_loads},
                                                  {"num_runs", *args.num_loads}};

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