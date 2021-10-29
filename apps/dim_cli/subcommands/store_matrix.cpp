#include "store_matrix.h"
#include "dim/io/format.h"

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <dim/io/format.h>
#include <dim/io/h5.h>
#include <dim/io/matrix_market.h>

#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

namespace {

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

auto load_config(const std::filesystem::path& config_path) {
    const auto node = YAML::LoadFile(config_path.string());
    return load_csr_props(node["csr"]);
}

} // namespace

auto store_matrix(const dim_cli::store_matrix_t& arguments) -> int {
    namespace h5 = dim::io::h5;

    using dim::io::formattable_bytes;
    using dim::io::h5::write_matlab_compatible;
    using dim::io::matrix_market::load_as_csr;
    using std::filesystem::file_size;

    spdlog::info("loading matrix in Matrix Market format from '{}', size: {:.3f}", arguments.input.string(),
                 formattable_bytes{file_size(arguments.input)});

    spdlog::stopwatch stopwatch{};
    const auto csr = load_as_csr<double>(arguments.input);
    spdlog::info("load and conversion to CSR took: {}s", stopwatch);

    auto file = h5::file_t::create(arguments.output, *arguments.append ? H5F_ACC_RDWR | H5F_ACC_CREAT : H5F_ACC_TRUNC);
    auto group = file.create_group(*arguments.group_name);

    spdlog::info("storing matrix as group '{}' to {}", *arguments.group_name, arguments.output.native());

    stopwatch.reset();
    write_matlab_compatible(group.get_id(), csr, arguments.config ? load_config(*arguments.config) : csr_props{});
    spdlog::info("storing CSR to HDF5 took: {}s", stopwatch);

    return 0;
}
