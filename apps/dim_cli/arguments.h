#ifndef APPS_DIM_CLI_ARGUMENTS
#define APPS_DIM_CLI_ARGUMENTS

#include <magic_enum.hpp>
#include <structopt/app.hpp>

#include <filesystem>
#include <optional>
#include <stdexcept>

#include <spdlog/common.h>

enum class download_format
{
    detect,
    archive,
    gzip
};

struct dim_cli
{
    using path = std::filesystem::path;
    using log_level_t = spdlog::level::level_enum;

    struct store_matrix_t : structopt::sub_command
    {
        enum class out_format
        {
            csr,
            csr5
        };

        path input;
        std::optional<path> output;
        std::optional<std::string> in_group_name{"A"};
        std::optional<std::string> group_name{"A"};
        std::optional<bool> append{false};
        std::optional<out_format> format{out_format::csr};

        std::optional<path> config{"config.yaml"};
    };
    store_matrix_t store_matrix;

    struct csr5_info_t : structopt::sub_command
    {
        path input;
        size_t row;
        std::optional<std::string> matrix_group{"A"};
    };
    csr5_info_t csr5_info;

    struct compare_results_t : structopt::sub_command
    {
        // file to load a matrix from.
        path input_file;
        // optionally, we can load from 2 different files, if empty, we just use the first file.
        std::optional<path> input_file_2;

        // group in HDF5 file from which to load LHS.
        std::optional<std::string> lhs_group{"/"};
        // dataset in HDF5 file from which to load LHS.
        std::optional<std::string> lhs_dataset{"Y"};

        // group in HDF5 file from which to load RHS.
        std::optional<std::string> rhs_group{"/"};
        // dataset in HDF5 file from which to load RHS.
        std::optional<std::string> rhs_dataset{"Y"};
    };
    compare_results_t compare_results;

    struct download_t : structopt::sub_command
    {
        std::string url;
        std::optional<path> destination_dir{std::filesystem::current_path()};
        std::optional<download_format> format = download_format::detect;
    };
    download_t download;

    struct generate_heatmap_t : structopt::sub_command
    {
        // file to load a matrix from.
        path input_file;
        // output file, defaulted to <input_file>.png
        std::optional<path> output_file;
        std::optional<std::string> group{"A"};
    };
    generate_heatmap_t generate_heatmap;

    std::optional<log_level_t> log_level{log_level_t::warn};
};

STRUCTOPT(dim_cli::compare_results_t, input_file, input_file_2, lhs_group, rhs_group);
STRUCTOPT(dim_cli::store_matrix_t, input, output, in_group_name, group_name, append, format, config);
STRUCTOPT(dim_cli::csr5_info_t, input, row, matrix_group);
STRUCTOPT(dim_cli::download_t, url, destination_dir, format);
STRUCTOPT(dim_cli::generate_heatmap_t, input_file, output_file, group);
STRUCTOPT(dim_cli, store_matrix, csr5_info, compare_results, download, generate_heatmap, log_level);

#endif /* APPS_DIM_CLI_ARGUMENTS */
