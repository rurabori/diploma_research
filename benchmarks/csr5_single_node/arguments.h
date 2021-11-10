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
    std::optional<bool> debug{false};
    std::optional<algorithm_t> algorithm{algorithm_t::cpu_avx2};
};

STRUCTOPT(arguments_t, input_file, debug, algorithm);

#endif /* BENCHMARKS_CSR5_SINGLE_NODE_ARGUMENTS */
