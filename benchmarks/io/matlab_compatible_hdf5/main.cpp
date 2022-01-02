#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <structopt/app.hpp>

#include <dim/io/h5.h>

#include <dim/simple_main.h>

#include <filesystem>

#include "version.h"

using dim::io::h5::read_matlab_compatible;

struct arguments_t
{
    std::filesystem::path matrix_path;
};
STRUCTOPT(arguments_t, matrix_path);

int main_impl(const arguments_t& args) {
    spdlog::stopwatch sw;
    const auto matrix = read_matlab_compatible(args.matrix_path, "A");
    spdlog::info("Loading took: {}s", sw);
    return 0;
}
DIM_MAIN(arguments_t);
