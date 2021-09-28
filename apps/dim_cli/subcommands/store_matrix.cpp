#include "store_matrix.h"

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <dim/io/h5.h>
#include <dim/io/matrix_market.h>

namespace {
H5::Group create_group_recurse(H5::Group base, std::string_view parts) {
    while (!parts.empty()) {
        // skip leading slashes
        parts.remove_prefix(parts.find_first_not_of('/'));

        // get the part between the slashes.
        const auto next_part = parts.substr(0, parts.find_first_of('/'));
        if (next_part.empty())
            break;

        base = base.createGroup(std::string{next_part});
        parts.remove_prefix(next_part.size());
    }

    return base;
}

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

void store_matrix(const dim_cli::store_matrix_t& arguments) {
    using dim::io::h5::write_matlab_compatible;
    using dim::io::matrix_market::load_as_csr;
    using std::filesystem::file_size;

    spdlog::info("loading matrix in Matrix Market format from '{}', size: {:.3f}MiB", arguments.input.string(),
                 static_cast<double>(file_size(arguments.input)) / 1'048'576);

    spdlog::stopwatch stopwatch{};
    const auto csr = load_as_csr<double>(arguments.input);
    spdlog::info("load and conversion to CSR took: {}s", stopwatch);

    H5::H5File file{arguments.output, *arguments.append ? H5F_ACC_RDWR | H5F_ACC_CREAT : H5F_ACC_TRUNC};

    auto matrix_group = create_group_recurse(file.openGroup("/"), *arguments.group_name);
    spdlog::info("storing matrix as group '{}' to {}", matrix_group.getObjName(), file.getFileName());

    stopwatch.reset();
    write_matlab_compatible(matrix_group, csr, create_matrix_storage_props(arguments));
    spdlog::info("storing CSR to HDF5 took: {}s", stopwatch);
}
