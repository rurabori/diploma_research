#include <H5Fpublic.h>
#include <H5Tpublic.h>
#include <dim/io/format.h>
#include <dim/io/h5.h>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <stdexcept>

#include "arguments.h"
#include "dim/io/h5/file.h"
#include "version.h"

namespace h5 = dim::io::h5;
using dim::mat::cache_aligned_vector;
using h5::load_csr5;

namespace {
auto ensure_dataset_ready(h5::file_view_t file, const std::string& name, bool overwrite) {
    if (!file.contains(name))
        return;

    if (overwrite)
        file.remove(name);

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

int main(int argc, char* argv[]) try {
    auto app = structopt::app(brr::app_info.full_name, brr::app_info.version);
    auto arguments = app.parse<arguments_t>(argc, argv);

    auto matrix = load_csr5(arguments.input_file, *arguments.matrix_group);

    auto dimensions = matrix.dimensions;

    auto x = cache_aligned_vector<double>(dimensions.cols, 1.);
    auto Y = cache_aligned_vector<double>(matrix.dimensions.rows, 0.);

    spdlog::stopwatch sw;
    matrix.spmv(dim::span{x}, dim::span{Y});
    spdlog::info("CSR5 SpMV took {}s", sw);

    if (arguments.output_file)
        output_result(arguments, Y);


    return 0;
} catch (const std::exception& e) {
    spdlog::critical("{}", e.what());
    return 1;
}