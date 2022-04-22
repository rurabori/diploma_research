#ifndef INCLUDE_DIM_IO_H5
#define INCLUDE_DIM_IO_H5

#include <hdf5.h>

#include <dim/mat/storage_formats.h>
#include <dim/memory/aligned_allocator.h>

#include <dim/io/file.h>
#include <dim/io/h5/attribute.h>
#include <dim/io/h5/dataset.h>
#include <dim/io/h5/dataspace.h>
#include <dim/io/h5/err.h>
#include <dim/io/h5/file.h>
#include <dim/io/h5/group.h>
#include <dim/io/h5/location.h>
#include <dim/io/h5/plist.h>
#include <dim/io/h5/type.h>

#include <cstddef>
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

    [[nodiscard]] auto as_checked_plist(size_t dataset_size) const -> plist_t;
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

        StorageTy retval;
        retval.resize(space.get_dim());
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

template<std::ranges::contiguous_range DataType, typename MemType = std::ranges::range_value_t<DataType>,
         h5::type_translator Translator = type_translator_t<MemType>>
auto write_dataset_2(location_view_t loc, const std::string& name, DataType&& data, plist_view_t prop_list) -> void {
    const auto dataspace = dataspace_t::create(std::size(data));

    loc.create_dataset(name, Translator::on_disk(), dataspace, plist_t::defaulted(), prop_list)
      .write(std::data(data), Translator::in_memory());
}

template<std::ranges::contiguous_range DataType>
auto write_dataset_2(location_view_t loc, const std::string& name, DataType&& data, const dataset_props_t& prop_list)
  -> void {
    write_dataset_2(loc, name, std::forward<DataType>(data), prop_list.as_checked_plist(std::size(data)));
}

template<std::integral DataType, h5::type_translator Translator = type_translator_t<DataType>>
auto write_attribute_2(group_view_t group, const std::string& name, DataType data) -> void {
    group.create_attribute(name, Translator::on_disk(), dataspace_t::create(H5S_SCALAR)).write(data);
}

template<template<typename> typename Storage>
void write_matlab_compatible(group_view_t group, const mat::csr<double, Storage>& matrix,
                             const matrix_storage_props_t& storage_props = {}) {
    write_dataset_2(group, "data", matrix.values, storage_props.values);
    write_dataset_2(group, "ir", matrix.col_indices, storage_props.col_idx);
    write_dataset_2(group, "jc", matrix.row_start_offsets, storage_props.row_start_offsets);

    write_attribute_2(group, "MATLAB_sparse", matrix.dimensions.cols);
}

template<template<typename> typename Storage = mat::cache_aligned_vector>
auto read_matlab_compatible(group_view_t group) -> mat::csr<double, Storage> {
    using retval_t = mat::csr<double, Storage>;
    using indices_t = typename retval_t::indices_t;

    auto&& row_start_offsets = group.open_dataset("jc").read<indices_t>();

    return retval_t{mat::dimensions_t{.rows = static_cast<uint32_t>(row_start_offsets.size() - 1),
                                      .cols = group.open_attribute("MATLAB_sparse")},
                    group.open_dataset("data"),   //
                    std::move(row_start_offsets), //
                    group.open_dataset("ir")};
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

auto load_csr5(const std::filesystem::path& path, const std::string& group_name) -> mat::csr5<double>;

auto read_vector(group_view_t group, const std::string& dataset_name) -> std::vector<double>;

auto is_hdf5(const std::filesystem::path& path) -> bool;

struct partial_identifier_t
{
    size_t idx;
    size_t total_count;
};

auto load_csr5_partial(group_view_t group, partial_identifier_t part) -> mat::csr5<double>;

auto load_csr_partial(group_view_t group, partial_identifier_t part) -> mat::csr_partial_t<>;

} // namespace dim::io::h5

#endif /* INCLUDE_DIM_IO_H5 */
