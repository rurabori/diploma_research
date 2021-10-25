#ifndef INCLUDE_DIM_IO_H5
#define INCLUDE_DIM_IO_H5

#include "dim/memory/aligned_allocator.h"
#include <H5Cpublic.h>
#include <H5Dpublic.h>
#include <H5Gpublic.h>
#include <H5Ppublic.h>
#include <H5Spublic.h>

#include <hdf5/H5Cpp.h>
#include <hdf5/H5DataType.h>
#include <hdf5/H5DcreatProp.h>
#include <hdf5/H5Group.h>
#include <hdf5/H5PredType.h>

#include <cstdint>
#include <dim/resource.h>

#include <dim/mat/storage_formats.h>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>

namespace dim::io::h5 {

template<const auto& DestroyFn>
struct hid_traits_t
{
    static void destroy(hid_t hid) noexcept { DestroyFn(hid); }
};

template<typename Traits>
using h5_resource_t = resource_t<hid_t, Traits>;

using plist_t = h5_resource_t<hid_traits_t<::H5Pclose>>;
using group_t = h5_resource_t<hid_traits_t<::H5Gclose>>;
using dataspace_t = h5_resource_t<hid_traits_t<::H5Sclose>>;
using datatype_t = h5_resource_t<hid_traits_t<::H5Tclose>>;
using dataset_t = h5_resource_t<hid_traits_t<::H5Dclose>>;
using attribute_t = h5_resource_t<hid_traits_t<::H5Aclose>>;

struct dataset_props_t
{
    std::optional<hsize_t> chunk_size{};
    std::optional<int> compression_level{};

    explicit operator plist_t() const;
};

namespace detail {
    template<typename /*std::ranges::contiguous_range*/ Ty>
    void write_dataset(const group_t& group, const std::string& name, const Ty& data, const datatype_t& input_type,
                       const datatype_t& storage_type, std::span<const hsize_t> dims, const plist_t& prop_list) {
        const auto dataspace = dataspace_t{::H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr)};

        const auto dataset = dataset_t{::H5Dcreate(group.get(), name.c_str(), storage_type.get(), dataspace.get(),
                                                   H5P_DEFAULT, prop_list.get(), H5P_DEFAULT)};

        // TODO: check return type.
        ::H5Dwrite(dataset.get(), input_type.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, std::data(data));
    }

    template<typename /*std::ranges::contiguous_range*/ Ty>
    void write_dataset(const group_t& group, const std::string& name, const Ty& data, const datatype_t& input_type,
                       const datatype_t& storage_type, const dataset_props_t& dataset_props) {
        hsize_t dims[1] = {std::size(data)};

        write_dataset(group, name, data, input_type, storage_type, dims, static_cast<plist_t>(dataset_props));
    }

    template<typename Ty>
    void write_scalar_datatype(const group_t& group, const std::string& name, const Ty& value,
                               const datatype_t& input_type, const datatype_t& storage_type) {
        const auto attribute = attribute_t{
          ::H5Acreate(group.get(), name.c_str(), storage_type.get(), H5S_SCALAR, H5P_DEFAULT, H5P_DEFAULT)};

        ::H5Awrite(attribute.get(), input_type.get(), &value);
    }

    template<typename StorageTy>
    auto read_dataset(const group_t& group, const std::string& name, const H5::DataType& storage_type,
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

struct matrix_storage_props_t
{
    dataset_props_t values;
    dataset_props_t col_idx;
    dataset_props_t row_start_offsets;
};

template<template<typename> typename Storage>
void write_matlab_compatible(H5::Group& group, const mat::csr<double, Storage>& matrix,
                             const matrix_storage_props_t& storage_props = {}) {
    using detail::write_dataset;
    using detail::write_scalar_datatype;

    write_dataset(group, "data", matrix.values, H5::PredType::NATIVE_DOUBLE, H5::PredType::IEEE_F64LE,
                  storage_props.values);
    write_dataset(group, "ir", matrix.col_indices, H5::PredType::NATIVE_UINT32, H5::PredType::STD_U64LE,
                  storage_props.col_idx);
    write_dataset(group, "jc", matrix.row_start_offsets, H5::PredType::NATIVE_UINT32, H5::PredType::STD_U64LE,
                  storage_props.row_start_offsets);
    write_scalar_datatype(group, "MATLAB_sparse", matrix.dimensions.cols, H5::PredType::NATIVE_UINT32,
                          H5::PredType::STD_U64LE);
}

template<template<typename> typename Storage = mat::cache_aligned_vector>
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

// TODO: better naming
auto store(H5::Group& group, const mat::csr5<double>& csr5) -> void;
auto load_csr5(const H5::Group& group) -> mat::csr5<double>;

auto read_vector(const H5::Group& group, const std::string& dataset_name) -> std::vector<double>;

} // namespace dim::io::h5

#endif /* INCLUDE_DIM_IO_H5 */
