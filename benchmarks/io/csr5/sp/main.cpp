#include <dim/bench/timed_section.h>
#include <dim/io/h5.h>
#include <dim/simple_main.h>

#include <structopt/app.hpp>

#include <nlohmann/json.hpp>

#include <fmt/chrono.h>
#include <fmt/ranges.h>

#include <fstream>

#include <omp.h>

namespace h5 = dim::io::h5;
namespace fs = std::filesystem;

struct arguments_t
{
    fs::path input_file;
    std::optional<size_t> num_loads{100};
    std::optional<std::string> group{"A"};
};
STRUCTOPT(arguments_t, input_file, num_loads, group);

using dim::bench::second;
using dim::bench::section;
using dim::bench::stopwatch;

int main_impl(const arguments_t& args) {
    auto total_time = second{};

    for (size_t i = 0; i < *args.num_loads; ++i)
        total_time += section([&] { auto mat = h5::load_csr5(args.input_file, *args.group); });

    std::ofstream{"stats.json"} << nlohmann::json{{"matrix", args.input_file.filename().string()},
                                                  {"total", total_time.count()},
                                                  {"avg", total_time.count() / *args.num_loads},
                                                  {"num_runs", *args.num_loads}};

    return 0;
}
DIM_MAIN(arguments_t);