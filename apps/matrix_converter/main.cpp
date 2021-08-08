

#include <dim/io/h5.h>
#include <dim/io/matrix_market.h>

#include <string>
#include <sys/types.h>
#include <tclap/Arg.h>
#include <tclap/CmdLine.h>
#include <tclap/SwitchArg.h>
#include <tclap/ValueArg.h>

#include <hdf5/H5Cpp.h>
#include <hdf5/H5DataSet.h>
#include <hdf5/H5DataSpace.h>
#include <hdf5/H5DataType.h>
#include <hdf5/H5File.h>
#include <hdf5/H5FloatType.h>
#include <hdf5/H5Fpublic.h>
#include <hdf5/H5Group.h>
#include <hdf5/H5PredType.h>
#include <hdf5/H5StrType.h>
#include <hdf5/H5Tpublic.h>
#include <hdf5/H5public.h>

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

template<std::ranges::contiguous_range Ty>
void write_dataset(H5::Group& group, const std::string& name, const Ty& data, const H5::DataType& input_type,
                   const H5::DataType& storage_type) {
    hsize_t dims[1] = {std::size(data)};

    H5::DataSpace dataspace{1, std::data(dims)};
    group.createDataSet(name, storage_type, dataspace).write(std::data(data), input_type);
}

int main(int argc, const char* argv[]) {
    using dim::io::matrix_market::load_as_csr;

    const auto arguments = arguments::from_main(argc, argv);
    const auto csr = load_as_csr<double>(arguments.input);

    H5::H5File file{arguments.output, arguments.append ? H5F_ACC_RDWR | H5F_ACC_CREAT : H5F_ACC_TRUNC};

    auto matrix_group = file.createGroup(arguments.group_name, 3);
    dim::io::h5::write_matlab_compatible(matrix_group, csr);

    return 0;
}