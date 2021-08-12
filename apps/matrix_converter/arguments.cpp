#include "arguments.h"
#include "version.h"

#include <magic_enum.hpp>
#include <spdlog/spdlog.h>
#include <tclap/HelpVisitor.h>

arguments arguments::from_main(int argc, const char* argv[]) {
    constexpr auto get_optional_path = [](auto&& arg) -> std::optional<path> {
        if (!arg.isSet())
            return std::nullopt;
        return arg.getValue();
    };

    constexpr auto get_log_level = [](std::string_view as_string) {
        if (auto opt = magic_enum::enum_cast<decltype(log_level)>(as_string); opt.has_value())
            return *opt;

        spdlog::warn("An invalid log level was provided, defaulting to warn.");
        return spdlog::level::warn;
    };

    TCLAP::CmdLine commandline{"Matrix Converter.", ' ', matrix_converter_VER};

    TCLAP::ValueArg<std::string> input_matrix_arg{"m", "input-matrix", "Path to input matrix", false, "", "string"};
    commandline.add(input_matrix_arg);

    TCLAP::ValueArg<std::string> input_vector_arg{"v", "input-vector", "Path to input vector", false, "", "string"};
    commandline.add(input_vector_arg);

    TCLAP::ValueArg<std::string> output_arg{"o", "output", "Path to output matrix", true, "", "string"};
    commandline.add(output_arg);

    TCLAP::ValueArg<std::string> matrix_group_name_arg{
      "g", "matix-group-name", "matrix group name in HDF5 file", false, "A", "string"};
    commandline.add(matrix_group_name_arg);

    TCLAP::ValueArg<std::string> vector_group_name_arg{
      "x", "vector-group-name", "vector group name in HDF5 file", false, "x", "string"};
    commandline.add(vector_group_name_arg);

    TCLAP::ValueArg<std::string> verbosity{"l",   "log-level", "logger level",
                                           false, "warn",      "{trace,debug,info,warn,error,critical}"};
    commandline.add(verbosity);

    TCLAP::SwitchArg append_arg{"a", "append-group", "append the group to an existing file.", false};
    commandline.add(append_arg);

    commandline.parse(argc, argv);

    if (!input_matrix_arg.isSet() && !input_vector_arg.isSet()) {
        throw std::invalid_argument{"At least one of input matrix or input vector has to be provided."};
    }

    return arguments{.matrix_input = get_optional_path(input_matrix_arg),
                     .vector_input = get_optional_path(input_vector_arg),
                     .output = output_arg.getValue(),
                     .matrix_group_name = matrix_group_name_arg.getValue(),
                     .vector_group_name = vector_group_name_arg.getValue(),
                     .append = append_arg.getValue(),
                     .log_level = get_log_level(verbosity.getValue())};
}
