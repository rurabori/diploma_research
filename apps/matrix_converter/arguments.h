#ifndef APPS_MATRIX_CONVERTER_ARGUMENTS
#define APPS_MATRIX_CONVERTER_ARGUMENTS

#include <structopt/app.hpp>

#include <filesystem>
#include <optional>
#include <stdexcept>

#include <spdlog/common.h>

struct dim_cli
{
    using path = std::filesystem::path;
    using log_level_t = spdlog::level::level_enum;

    struct store_matrix_t : structopt::sub_command
    {
        path input;
        path output;
        std::optional<std::string> group_name{"A"};
        std::optional<bool> append{false};
    };
    store_matrix_t store_matrix;

    std::optional<log_level_t> log_level{log_level_t::warn};
};

STRUCTOPT(dim_cli::store_matrix_t, input, output);
STRUCTOPT(dim_cli, store_matrix, log_level);

#endif /* APPS_MATRIX_CONVERTER_ARGUMENTS */
