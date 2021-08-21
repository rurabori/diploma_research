#ifndef INCLUDE_DIM_IO_H5
#define INCLUDE_DIM_IO_H5

#include "dim/memory/aligned_allocator.h"
#include <cstdint>
#include <hdf5/H5Cpp.h>
#include <hdf5/H5DataType.h>
#include <hdf5/H5Group.h>
#include <hdf5/H5PredType.h>

#include <dim/mat/storage_formats.h>
#include <stdexcept>

namespace dim::io::h5 {

namespace detail {
    template<typename /*std::ranges::contiguous_range*/ Ty>
    void write_dataset(H5::Group& group, const std::string& name, const Ty& data, const H5::DataType& input_type,
                       const H5::DataType& storage_type) {
        hsize_t dims[1] = {std::size(data)};

        H5::DataSpace dataspace{1, std::data(dims)};
        group.createDataSet(name, storage_type, dataspace).write(std::data(data), input_type);
    }

    template<typename Ty>
    void write_scalar_datatype(H5::Group& group, const std::string& name, const Ty& value,
                               const H5::DataType& input_type, const H5::DataType& storage_type) {
        group.createAttribute(name, storage_type, H5::DataSpace{H5S_SCALAR}).write(input_type, &value);
    }

    template<typename StorageTy>
    auto read_dataset(const H5::Group& group, const std::string& name, const H5::DataType& storage_type,
                      const H5::DataType& memory_type) {
        const auto dataset = group.openDataSet(name);
        if (dataset.getDataType() != storage_type)
            throw std::invalid_argument{"Expecting a different storage type."};

        const auto space = dataset.getSpace();
        if (space.getSimpleExtentNdims() != 1)
            throw std::invalid_argument{"Expecting only a single dimension."};

        hsize_t size{};
        space.getSimpleExtentDims(&size, nullptr);

        StorageTy retval(size);
        dataset.read(std::data(retval), memory_type);

        return retval;
    }

    template<typename Ty>
    auto read_scalar_datatype(const H5::Group& group, const std::string& name, const H5::DataType& storage_type,
                              const H5::DataType& memory_type) {
        const auto attr = group.openAttribute(name);
        if (attr.getDataType() != storage_type)
            throw std::invalid_argument{"Expecting a different storage type."};

        Ty retval{};
        attr.read(memory_type, &retval);

        return retval;
    }

} // namespace detail

template<template<typename> typename Storage>
void write_matlab_compatible(H5::Group& group, const mat::csr<double, Storage>& matrix) {
    using detail::write_dataset;
    using detail::write_scalar_datatype;

    write_dataset(group, "data", matrix.values, H5::PredType::NATIVE_DOUBLE, H5::PredType::IEEE_F64LE);
    write_dataset(group, "ir", matrix.col_indices, H5::PredType::NATIVE_UINT32, H5::PredType::STD_U64LE);
    write_dataset(group, "jc", matrix.row_start_offsets, H5::PredType::NATIVE_UINT32, H5::PredType::STD_U64LE);
    write_scalar_datatype(group, "MATLAB_sparse", matrix.dimensions.cols, H5::PredType::NATIVE_UINT32,
                          H5::PredType::STD_U64LE);
}

template<template<typename> typename Storage = memory::cache_aligned_vector>
auto read_matlab_compatible(const H5::Group& group) -> mat::csr<double, Storage> {
    using retval_t = mat::csr<double, Storage>;
    using indices_t = typename retval_t::indices_t;
    using values_t = typename retval_t::values_t;
    using detail::read_dataset;
    using detail::read_scalar_datatype;

    auto values = read_dataset<values_t>(group, "data", H5::PredType::IEEE_F64LE, H5::PredType::NATIVE_DOUBLE);
    auto col_indices = read_dataset<indices_t>(group, "ir", H5::PredType::STD_U64LE, H5::PredType::NATIVE_UINT32);
    auto row_start_offsets = read_dataset<indices_t>(group, "jc", H5::PredType::STD_U64LE, H5::PredType::NATIVE_UINT32);
    const auto num_cols
      = read_scalar_datatype<uint32_t>(group, "MATLAB_sparse", H5::PredType::STD_U64LE, H5::PredType::NATIVE_UINT32);

    retval_t retval{mat::dimensions_t{static_cast<uint32_t>(row_start_offsets.size() - 1), num_cols}, std::move(values),
                    std::move(row_start_offsets), std::move(col_indices)};

    return retval;
}

auto read_vector(const H5::Group& group, const std::string& dataset_name) -> std::vector<double>;

} // namespace dim::io::h5

#endif /* INCLUDE_DIM_IO_H5 */
