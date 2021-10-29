#include "store_matrix.h"
#include "arguments.h"
#include "dim/io/format.h"

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <dim/io/format.h>
#include <dim/io/h5.h>
#include <dim/io/matrix_market.h>

#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

namespace {

using csr5_props = dim::io::h5::csr5_storage_props_t;
using csr_props = dim::io::h5::matrix_storage_props_t;
using csr_dataset_props = dim::io::h5::dataset_props_t;

template<typename Ty>
auto maybe_load(const YAML::Node& node) -> std::optional<Ty> {
    if (node.IsNull())
        return std::nullopt;

    return node.as<Ty>();
}

auto load_dataset_props(const YAML::Node& node) -> csr_dataset_props {
    return csr_dataset_props{.chunk_size = maybe_load<hsize_t>(node["chunk_size"]),
                             .compression_level = maybe_load<uint32_t>(node["compression_level"])};
}

auto load_csr_props(const YAML::Node& node) -> dim::io::h5::matrix_storage_props_t {
    if (!node.IsMap())
        throw std::runtime_error{"Dataset props node must be an object."};

    return csr_props{.values = load_dataset_props(node["values"]),
                     .col_idx = load_dataset_props(node["col_idx"]),
                     .row_start_offsets = load_dataset_props(node["row_idx"])};
}

auto load_csr5_props(const YAML::Node& node) -> csr5_props {
    if (!node.IsMap())
        throw std::runtime_error{"Dataset props node must be an object."};

    return csr5_props{
      .vals = load_dataset_props(node["vals"]),
      .col_idx = load_dataset_props(node["col_idx"]),
      .row_ptr = load_dataset_props(node["row_ptr"]),
      .tile_ptr = load_dataset_props(node["tile_ptr"]),
      .tile_desc = load_dataset_props(node["tile_desc"]),
      .tile_desc_offset = load_dataset_props(node["tile_desc_offset"]),
      .tile_desc_offset_ptr = load_dataset_props(node["tile_desc_offset_ptr"]),
    };
}

struct config
{
    csr_props csr;
    csr5_props csr5;
};

auto load_config(const std::filesystem::path& config_path) {
    const auto node = YAML::LoadFile(config_path.string());

    return config{.csr = load_csr_props(node["csr"]), .csr5 = load_csr5_props(node["csr5"])};
}

auto load_as_csr(const dim_cli::store_matrix_t& arguments) -> dim::mat::csr<double> {
    namespace h5 = dim::io::h5;
    namespace mm = dim::io::matrix_market;

    if (!h5::is_hdf5(arguments.input))
        return mm::load_as_csr<double>(arguments.input);

    return h5::read_matlab_compatible(arguments.input, *arguments.in_group_name);
}

auto convert_to_hdf5(dim::mat::csr<double>&& csr) {
    spdlog::stopwatch stopwatch{};
    auto&& csr5 = dim::mat::csr5<>::from_csr(std::move(csr));
    spdlog::info("conversion to CSR5 took: {}s", stopwatch);

    return std::forward<decltype(csr5)>(csr5);
}

auto store(const dim_cli::store_matrix_t& arguments, dim::mat::csr<double>&& matrix) {
    using format_t = dim_cli::store_matrix_t::out_format;
    namespace h5 = dim::io::h5;

    auto file = h5::file_t::create(arguments.output, *arguments.append ? H5F_ACC_RDWR | H5F_ACC_CREAT : H5F_ACC_TRUNC);
    auto group = file.create_group(*arguments.group_name);

    const auto config = load_config(*arguments.config);

    switch (*arguments.format) {
        case format_t::csr:
            h5::write_matlab_compatible(group, matrix, config.csr);
            break;
        case format_t::csr5:
            h5::store(group, convert_to_hdf5(std::move(matrix)), config.csr5);
            break;
    }
}

} // namespace

auto store_matrix(const dim_cli::store_matrix_t& arguments) -> int {
    using dim::io::formattable_bytes;
    using std::filesystem::file_size;

    spdlog::info("loading matrix from '{}', size: {:.3f}", arguments.input.string(),
                 formattable_bytes{file_size(arguments.input)});

    spdlog::stopwatch stopwatch{};
    auto&& csr = load_as_csr(arguments);
    spdlog::info("load and conversion to CSR took: {}s", stopwatch);

    spdlog::info("storing matrix as group '{}' to {}", *arguments.group_name, arguments.output.native());

    stopwatch.reset();
    store(arguments, std::move(csr));
    spdlog::info("storing as HDF5 took: {}s", stopwatch);

    return 0;
}
