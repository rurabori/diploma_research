#ifndef INCLUDE_DIM_IO_H5
#define INCLUDE_DIM_IO_H5

#include "dim/io/file.h"

#include <hdf5.h>

#include <cstddef>
#include <dim/mat/storage_formats.h>
#include <dim/memory/aligned_allocator.h>
#include <dim/resource.h>

#include <dim/io/h5/attribute.h>
#include <dim/io/h5/dataset.h>
#include <dim/io/h5/dataspace.h>
#include <dim/io/h5/err.h>
#include <dim/io/h5/file.h>
#include <dim/io/h5/group.h>
#include <dim/io/h5/location.h>
#include <dim/io/h5/plist.h>
#include <dim/io/h5/type.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>

namespace dim::io::h5 {

struct dataset_props_t
{
    std::optional<hsize_t> chunk_size{};
    std::optional<uint32_t> compression_level{};

    explicit operator plist_t() const;
};

namespace detail {
    template<typename /*std::ranges::contiguous_range*/ Ty>
    void write_dataset(location_view_t group, const std::string& name, const Ty& data, type_view_t input_type,
                       type_view_t storage_type, std::span<const hsize_t> dims, plist_view_t prop_list) {
        auto dataspace = dataspace_t::create(dims);
        auto dataset = group.create_dataset(name, storage_type, dataspace, plist_t::defaulted(), prop_list);
        dataset.write(std::data(data), input_type);
    }

    template<typename /*std::ranges::contiguous_range*/ Ty>
    void write_dataset(location_view_t group, const std::string& name, const Ty& data, type_view_t input_type,
                       type_view_t storage_type, const dataset_props_t& dataset_props) {
        hsize_t dims[1] = {std::size(data)};

        write_dataset(group, name, data, type_view_t{input_type}, type_view_t{storage_type}, dims,
                      static_cast<plist_t>(dataset_props));
    }

    template<typename Ty>
    void write_scalar_datatype(group_view_t group, const std::string& name, const Ty& value, type_view_t input_type,
                               type_view_t storage_type) {
        auto space = dataspace_t::create(H5S_SCALAR);
        auto attribute = group.create_attribute(name, storage_type, space);

        attribute.write(&value, input_type);
    }

    template<typename StorageTy>
    auto read_dataset(group_view_t group, const std::string& name, type_view_t storage_type, type_view_t memory_type) {
        auto dataset = group.open_dataset(name);

        if (dataset.get_type() != storage_type)
            throw std::invalid_argument{"Expecting a different storage type."};

        auto space = dataset.get_dataspace();

        StorageTy retval(space.get_dim());
        dataset.read(std::data(retval), memory_type);

        return retval;
    }

    template<typename Ty>
    auto read_scalar_datatype(group_view_t group, const std::string& name, type_view_t storage_type,
                              type_view_t memory_type) {
        auto attr = group.open_attribute(name);

        if (attr.get_type() != storage_type)
            throw std::invalid_argument{"Expecting a different storage type."};

        Ty retval{};
        attr.read(&retval, memory_type);

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
void write_matlab_compatible(group_view_t group, const mat::csr<double, Storage>& matrix,
                             const matrix_storage_props_t& storage_props = {}) {
    using detail::write_dataset;
    using detail::write_scalar_datatype;

    write_dataset(group, "data", matrix.values, H5T_NATIVE_DOUBLE, H5T_IEEE_F64LE, storage_props.values);
    write_dataset(group, "ir", matrix.col_indices, H5T_NATIVE_UINT32, H5T_STD_U64LE, storage_props.col_idx);
    write_dataset(group, "jc", matrix.row_start_offsets, H5T_NATIVE_UINT32, H5T_STD_U64LE,
                  storage_props.row_start_offsets);
    write_scalar_datatype(group_view_t{group}, "MATLAB_sparse", matrix.dimensions.cols, H5T_NATIVE_UINT32,
                          H5T_STD_U64LE);
}

template<template<typename> typename Storage = mat::cache_aligned_vector>
auto read_matlab_compatible(group_view_t group) -> mat::csr<double, Storage> {
    using retval_t = mat::csr<double, Storage>;
    using indices_t = typename retval_t::indices_t;
    using values_t = typename retval_t::values_t;
    using detail::read_dataset;
    using detail::read_scalar_datatype;

    auto values = read_dataset<values_t>(group, "data", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE);
    auto col_indices = read_dataset<indices_t>(group, "ir", H5T_STD_U64LE, H5T_NATIVE_UINT32);
    auto row_start_offsets = read_dataset<indices_t>(group, "jc", H5T_STD_U64LE, H5T_NATIVE_UINT32);
    const auto num_cols = read_scalar_datatype<uint32_t>(group, "MATLAB_sparse", H5T_STD_U64LE, H5T_NATIVE_UINT32);

    retval_t retval{mat::dimensions_t{static_cast<uint32_t>(row_start_offsets.size() - 1), num_cols}, std::move(values),
                    std::move(row_start_offsets), std::move(col_indices)};

    return retval;
}

template<template<typename> typename Storage = mat::cache_aligned_vector>
auto read_matlab_compatible(const std::filesystem::path& path, const std::string& group_name)
  -> mat::csr<double, Storage> {
    auto file = h5::file_t::open(path, H5F_ACC_RDONLY);
    return read_matlab_compatible<Storage>(file.open_group(group_name));
}

struct csr5_storage_props_t
{
    dataset_props_t vals;
    dataset_props_t col_idx;
    dataset_props_t row_ptr;
    dataset_props_t tile_ptr;
    dataset_props_t tile_desc;
    dataset_props_t tile_desc_offset;
    dataset_props_t tile_desc_offset_ptr;
};

// TODO: better naming
auto store(group_view_t group, const mat::csr5<double>& csr5, const csr5_storage_props_t& props = {}) -> void;
auto load_csr5(group_view_t group) -> mat::csr5<double>;

auto read_vector(group_view_t group, const std::string& dataset_name) -> std::vector<double>;

auto is_hdf5(const std::filesystem::path& path) -> bool;

auto load_csr5_partial(group_view_t group, size_t part, size_t num_parts) -> mat::csr5<double>;

} // namespace dim::io::h5

#endif /* INCLUDE_DIM_IO_H5 */
