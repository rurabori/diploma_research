

#include <dim/io/h5.h>
#include <dim/io/matrix_market.h>

#include <string>
#include <sys/types.h>
#include <tclap/Arg.h>
#include <tclap/CmdLine.h>
#include <tclap/SwitchArg.h>
#include <tclap/ValueArg.h>

#include <hdf5/H5Cpp.h>
#include <hdf5/H5File.h>

#include <concepts>
#include <filesystem>
#include <ranges>

#include "version.h"

struct arguments
{
    std::filesystem::path input{};
    std::filesystem::path output{};
    std::string group_name{};
    bool append{false};

    static arguments from_main(int argc, const char* argv[]) {
        TCLAP::CmdLine commandline{"Matrix Converter.", ' ', matrix_converter_VER};

        TCLAP::ValueArg<std::string> input_arg{"i", "input", "Path to input matrix", true, "", "string"};
        commandline.add(input_arg);

        TCLAP::ValueArg<std::string> output_arg{"o", "output", "Path to output matrix", true, "", "string"};
        commandline.add(output_arg);

        TCLAP::ValueArg<std::string> group_name_arg{"g",   "group-name", "matrix group name in HDF5 file",
                                                    false, "A",          "string"};
        commandline.add(group_name_arg);

        TCLAP::SwitchArg append_arg{"a", "append-group", "append the group to an existing file.", false};
        commandline.add(append_arg);

        commandline.parse(argc, argv);

        return arguments{.input = input_arg.getValue(),
                         .output = output_arg.getValue(),
                         .group_name = group_name_arg.getValue(),
                         .append = append_arg.getValue()};
    }
};

int main(int argc, const char* argv[]) {
    using dim::io::matrix_market::load_as_csr;

    const auto arguments = arguments::from_main(argc, argv);
    const auto csr = load_as_csr<double>(arguments.input);

    H5::H5File file{arguments.output, arguments.append ? H5F_ACC_RDWR | H5F_ACC_CREAT : H5F_ACC_TRUNC};

    auto matrix_group = file.createGroup(arguments.group_name, 3);
    dim::io::h5::write_matlab_compatible(matrix_group, csr);

    return 0;
}