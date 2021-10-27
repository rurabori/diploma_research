#include "store_matrix.h"
#include "dim/io/format.h"

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <dim/io/format.h>
#include <dim/io/h5.h>
#include <dim/io/matrix_market.h>

namespace {

auto create_matrix_storage_props(const dim_cli::store_matrix_t& arguments) {
    using dim::io::h5::matrix_storage_props_t;
    constexpr auto as_hsize = [](const auto& val) -> std::optional<hsize_t> {
        if (!val)
            return std::nullopt;
        return static_cast<hsize_t>(*val);
    };

    return matrix_storage_props_t{.values = {.chunk_size = as_hsize(arguments.values_chunk_size),
                                             .compression_level = arguments.values_compression},
                                  .col_idx = {.chunk_size = as_hsize(arguments.col_idx_chunk_size),
                                              .compression_level = arguments.col_idx_compression},
                                  .row_start_offsets = {.chunk_size = as_hsize(arguments.row_start_offsets_chunk_size),
                                                        .compression_level = arguments.row_start_offsets_compression}};
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

    auto file = h5::file_t{::H5Fcreate(arguments.output.c_str(),
                                       *arguments.append ? H5F_ACC_RDWR | H5F_ACC_CREAT : H5F_ACC_TRUNC, H5P_DEFAULT,
                                       H5P_DEFAULT)};
    auto group
      = h5::group_t{::H5Gcreate(file.get(), arguments.group_name->c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)};

    spdlog::info("storing matrix as group '{}' to {}", *arguments.group_name, arguments.output.native());

    stopwatch.reset();
    write_matlab_compatible(group.get(), csr, create_matrix_storage_props(arguments));
    spdlog::info("storing CSR to HDF5 took: {}s", stopwatch);

    return 0;
}
