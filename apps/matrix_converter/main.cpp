#include <dim/io/h5.h>
#include <dim/io/matrix_market.h>

#include <filesystem>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>

#include <hdf5/H5Cpp.h>
#include <hdf5/H5Exception.h>
#include <hdf5/H5File.h>
#include <hdf5/H5Group.h>

#include <spdlog/common.h>
#include <spdlog/logger.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <string_view>

#include "arguments.h"

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

void store_matrix(const arguments& arguments, H5::H5File& file) {
    using dim::io::matrix_market::load_as_csr;
    using std::filesystem::file_size;

    if (!arguments.matrix_input)
        return;

    spdlog::info("loading matrix in Matrix Market format from '{}', size: {:.3f}MiB", arguments.matrix_input->string(),
                 static_cast<double>(file_size(*arguments.matrix_input)) / 1'048'576);

    spdlog::stopwatch stopwatch{};
    const auto csr = load_as_csr<double>(*arguments.matrix_input);
    spdlog::info("load and conversion to CSR took: {}s", stopwatch);

    auto matrix_group = create_group_recurse(file.openGroup("/"), arguments.matrix_group_name);
    spdlog::info("storing matrix as group '{}' to {}", matrix_group.getObjName(), file.getFileName());

    stopwatch.reset();
    dim::io::h5::write_matlab_compatible(matrix_group, csr);
    spdlog::info("storing CSR to HDF5 took: {}s", stopwatch);
}

int main(int argc, const char* argv[]) try {
    const auto arguments = arguments::from_main(argc, argv);
    spdlog::set_level(arguments.log_level);

    H5::H5File file{arguments.output, arguments.append ? H5F_ACC_RDWR | H5F_ACC_CREAT : H5F_ACC_TRUNC};
    store_matrix(arguments, file);

    return 0;
} catch (const H5::Exception& e) {
    spdlog::critical("HDF5 error @{}: '{}'", e.getFuncName(), e.getDetailMsg());
    return 1;
} catch (const std::exception& e) {
    spdlog::critical("exception thrown: {}", e.what());
    return 2;
}