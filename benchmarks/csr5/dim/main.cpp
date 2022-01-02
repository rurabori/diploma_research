#include <dim/bench/timed_section.h>
#include <dim/io/format.h>
#include <dim/io/h5.h>
#include <dim/simple_main.h>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <fmt/chrono.h>
#include <fmt/ranges.h>

#include <stdexcept>

#include "arguments.h"
#include "version.h"

namespace h5 = dim::io::h5;
using dim::mat::cache_aligned_vector;
using h5::load_csr5;

namespace {
auto ensure_dataset_ready(h5::file_view_t file, const std::string& name, bool overwrite) {
    if (!file.contains(name))
        return;

    if (overwrite) {
        file.remove(name);
        return;
    }

    throw std::runtime_error{fmt::format(
      "outfile already contains group {}, add overwrite flag to commandline if you really want to overwrite it", name)};
}

auto output_result(const arguments_t arguments, std::span<double> result) {
    auto outfile = h5::file_t::open(*arguments.output_file, H5F_ACC_CREAT | H5F_ACC_RDWR);

    ensure_dataset_ready(outfile, *arguments.vector_dataset, *arguments.overwrite);
    auto dataset
      = outfile.create_dataset(*arguments.vector_dataset, H5T_IEEE_F64LE, h5::dataspace_t::create(result.size()));

    dataset.write(result);
}

} // namespace

int main_impl(const arguments_t& arguments) {
    spdlog::set_default_logger(spdlog::stdout_color_mt("csr5_single_node"));

    dim::bench::stopwatch sw;
    auto matrix = load_csr5(arguments.input_file, *arguments.matrix_group);
    spdlog::info("CSR5 loading took {}s", sw.elapsed());

    auto dimensions = matrix.dimensions;

    auto x = cache_aligned_vector<double>(dimensions.cols, 1.);
    auto Y = cache_aligned_vector<double>(matrix.dimensions.rows, 0.);
    auto calibrator = matrix.allocate_calibrator();

    auto run_times = std::vector<decltype(sw.elapsed())>(*arguments.num_runs);

    auto total_time = dim::bench::second{0};

    for (size_t i = 0; i < *arguments.num_runs; ++i)
        total_time += dim::bench::section([&] { matrix.spmv({.x = x, .y = Y, .calibrator = calibrator}); });

    spdlog::info("CSR5 SpMV took {} on average", total_time / *arguments.num_runs);

    if (arguments.output_file)
        output_result(arguments, Y);

    return 0;
}
DIM_MAIN(arguments_t);