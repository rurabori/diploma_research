#ifndef BENCHMARKS_CSR5_SINGLE_NODE_ARGUMENTS
#define BENCHMARKS_CSR5_SINGLE_NODE_ARGUMENTS

#include <magic_enum.hpp>
#include <structopt/app.hpp>

#include <filesystem>

struct arguments_t
{
    enum class algorithm_t
    {
        cpu_sequential,
        cpu_avx2
    };

    using path = std::filesystem::path;

    path input_file;

    std::optional<size_t> num_runs{1};
    std::optional<std::string> matrix_group{"A"};
    std::optional<path> output_file;
    std::optional<std::string> vector_dataset{"Y"};
    std::optional<bool> overwrite{false};
};

STRUCTOPT(arguments_t, input_file, num_runs, matrix_group, output_file, vector_dataset, overwrite);

#endif /* BENCHMARKS_CSR5_SINGLE_NODE_ARGUMENTS */
