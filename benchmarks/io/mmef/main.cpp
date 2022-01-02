#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <structopt/app.hpp>

#include <dim/io/matrix_market.h>

#include <dim/simple_main.h>
#include <filesystem>

using dim::io::matrix_market::load_as_csr;

struct arguments_t
{
    std::filesystem::path matrix_path;
};
STRUCTOPT(arguments_t, matrix_path);

int main_impl(const arguments_t& args) {
    spdlog::stopwatch sw;
    const auto matrix = load_as_csr<double>(args.matrix_path);
    spdlog::info("Loading took: {}s", sw);

    return 0;
}
DIM_MAIN(arguments_t);
