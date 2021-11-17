#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <structopt/app.hpp>

#include <dim/io/matrix_market.h>

#include <filesystem>

#include "version.h"

using dim::io::matrix_market::load_as_csr;

struct arguments_t
{
    std::filesystem::path matrix_path;
};
STRUCTOPT(arguments_t, matrix_path);

int main(int argc, char* argv[]) try {
    const auto args = structopt::app(brr::app_info.name, brr::app_info.version).parse<arguments_t>(argc, argv);

    spdlog::stopwatch sw;
    const auto matrix = load_as_csr<double>(args.matrix_path);
    spdlog::info("Loading took: {}s", sw);
    return 0;
} catch (const std::exception& e) {
    spdlog::critical("Exception thrown: {}", e.what());
    return 1;
}
