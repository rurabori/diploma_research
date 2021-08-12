#ifndef APPS_MATRIX_CONVERTER_ARGUMENTS
#define APPS_MATRIX_CONVERTER_ARGUMENTS

#include <tclap/Arg.h>
#include <tclap/ArgException.h>
#include <tclap/CmdLine.h>
#include <tclap/SwitchArg.h>
#include <tclap/ValueArg.h>

#include <filesystem>
#include <optional>
#include <stdexcept>

#include <spdlog/common.h>

struct arguments
{
    using path = std::filesystem::path;

    std::optional<path> matrix_input{};
    std::optional<path> vector_input{};
    std::filesystem::path output{};
    std::string matrix_group_name{};
    std::string vector_group_name{};
    bool append{false};
    spdlog::level::level_enum log_level{};

    static arguments from_main(int argc, const char* argv[]);
};

#endif /* APPS_MATRIX_CONVERTER_ARGUMENTS */
