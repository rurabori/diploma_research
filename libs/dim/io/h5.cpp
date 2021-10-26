#include <H5Ppublic.h>
#include <H5version.h>
#include <dim/io/h5.h>

namespace dim::io::h5 {

dataset_props_t::operator plist_t() const {
    plist_t prop_list{::H5Pcreate(H5P_DATASET_CREATE)};
    if (chunk_size)
        ::H5Pset_chunk(prop_list.get(), 1, std::addressof(*chunk_size));

    if (compression_level)
        ::H5Pset_deflate(prop_list.get(), *compression_level);

    return prop_list;
}

struct csr5_storage_props_t
{
    dataset_props_t vals;
    dataset_props_t col_idx;
    dataset_props_t row_ptr;
    dataset_props_t tile_ptr;
    dataset_props_t tile_desc;
    dataset_props_t tile_desc_offset_ptr;
    dataset_props_t tile_desc_offset;
};

struct tile_types_t
{
    static constexpr hsize_t array_size = 4;

    datatype_t on_disk{::H5Tarray_create(H5T_NATIVE_UINT32, 1, &array_size)};
    datatype_t in_memory{::H5Tarray_create(H5T_STD_U32LE, 1, &array_size)};

    // TODO: make const
    static auto create() -> tile_types_t& {
        static tile_types_t tile_types;
        return tile_types;
    }
};

auto store(hid_t group, const dim::mat::csr5<double>& csr5) -> void {
    using detail::write_dataset;
    using detail::write_scalar_datatype;
    write_scalar_datatype(group, "column_count", csr5.dimensions.cols, H5T_NATIVE_UINT32, H5T_STD_U32LE);

    const auto props = csr5_storage_props_t{};
    auto&& tile_types = tile_types_t::create();

    write_dataset(group, "vals", csr5.vals, H5T_NATIVE_DOUBLE, H5T_IEEE_F64LE, props.vals);
    write_dataset(group, "col_idx", csr5.col_idx, H5T_NATIVE_UINT32, H5T_STD_U32LE, props.col_idx);
    write_dataset(group, "row_ptr", csr5.row_ptr, H5T_NATIVE_UINT32, H5T_STD_U32LE, props.row_ptr);
    write_dataset(group, "tile_ptr", csr5.tile_ptr, H5T_NATIVE_UINT32, H5T_STD_U32LE, props.tile_ptr);
    write_dataset(group, "tile_desc", csr5.tile_desc, tile_types.in_memory.get(), tile_types.on_disk.get(),
                  props.tile_desc);
    write_dataset(group, "tile_desc_offset_ptr", csr5.tile_desc_offset_ptr, H5T_NATIVE_UINT32, H5T_STD_U32LE,
                  props.tile_desc_offset_ptr);
    write_dataset(group, "tile_desc_offset", csr5.tile_desc_offset, H5T_NATIVE_UINT32, H5T_STD_U32LE,
                  props.tile_desc_offset);
}

auto load_csr5(hid_t group) -> dim::mat::csr5<double> {
    using retval_t = mat::csr5<double>;
    using detail::read_dataset;
    using detail::read_scalar_datatype;

    auto&& tile_types = tile_types_t::create();

    const auto num_cols = read_scalar_datatype<uint32_t>(group, "column_count", H5T_STD_U32LE, H5T_NATIVE_UINT32);

    retval_t retval{
      .vals = read_dataset<decltype(retval_t::vals)>(group, "vals", H5T_NATIVE_DOUBLE, H5T_IEEE_F64LE),
      .col_idx = read_dataset<decltype(retval_t::col_idx)>(group, "col_idx", H5T_NATIVE_UINT32, H5T_STD_U32LE),
      .row_ptr = read_dataset<decltype(retval_t::row_ptr)>(group, "row_ptr", H5T_NATIVE_UINT32, H5T_STD_U32LE),
      .tile_ptr = read_dataset<decltype(retval_t::tile_ptr)>(group, "tile_ptr", H5T_NATIVE_UINT32, H5T_STD_U32LE),
      .tile_desc = read_dataset<decltype(retval_t::tile_desc)>(group, "tile_desc", tile_types.in_memory.get(),
                                                               tile_types.on_disk.get()),
      .tile_desc_offset_ptr = read_dataset<decltype(retval_t::tile_desc_offset_ptr)>(group, "tile_desc_offset_ptr",
                                                                                     H5T_NATIVE_UINT32, H5T_STD_U32LE),
      .tile_desc_offset = read_dataset<decltype(retval_t::tile_desc_offset)>(group, "tile_desc_offset",
                                                                             H5T_NATIVE_UINT32, H5T_STD_U32LE)};

    retval.tile_count = retval.tile_desc.size();
    retval.dimensions = {.rows = static_cast<uint32_t>(retval.row_ptr.size() - 1), .cols = num_cols};

    return retval;
}

auto read_vector(hid_t group, const std::string& dataset_name) -> std::vector<double> {
    return detail::read_dataset<std::vector<double>>(group, dataset_name, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE);
}

} // namespace dim::io::h5