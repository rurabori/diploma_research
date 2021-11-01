#include "dim/io/h5/dataset.h"
#include "dim/io/h5/dataspace.h"
#include "dim/io/h5/type.h"
#include "dim/mat/storage_formats/base.h"
#include <H5Ppublic.h>
#include <H5version.h>
#include <dim/io/h5.h>

namespace dim::io::h5 {

dataset_props_t::operator plist_t() const {
    auto result = plist_t::create(H5P_DATASET_CREATE);
    if (chunk_size)
        ::H5Pset_chunk(result.get_id(), 1, std::addressof(*chunk_size));

    if (compression_level)
        ::H5Pset_deflate(result.get_id(), *compression_level);

    return result;
}

struct tile_types_t
{
    static constexpr hsize_t array_size = 4;

    type_t on_disk{type_t::create_array(H5T_NATIVE_UINT32, array_size)};
    type_t in_memory{type_t::create_array(H5T_STD_U32LE, array_size)};

    // TODO: make const
    static auto create() -> tile_types_t& {
        static tile_types_t tile_types;
        return tile_types;
    }
};

auto store(group_view_t group, const dim::mat::csr5<double>& csr5, const csr5_storage_props_t& props) -> void {
    using detail::write_dataset;
    using detail::write_scalar_datatype;
    write_scalar_datatype(group_view_t{group}, "column_count", csr5.dimensions.cols, H5T_NATIVE_UINT32, H5T_STD_U32LE);

    auto&& tile_types = tile_types_t::create();

    write_dataset(group, "vals", csr5.vals, H5T_NATIVE_DOUBLE, H5T_IEEE_F64LE, props.vals);
    write_dataset(group, "col_idx", csr5.col_idx, H5T_NATIVE_UINT32, H5T_STD_U32LE, props.col_idx);
    write_dataset(group, "row_ptr", csr5.row_ptr, H5T_NATIVE_UINT32, H5T_STD_U32LE, props.row_ptr);
    write_dataset(group, "tile_ptr", csr5.tile_ptr, H5T_NATIVE_UINT32, H5T_STD_U32LE, props.tile_ptr);
    write_dataset(group, "tile_desc", csr5.tile_desc, tile_types.in_memory, tile_types.on_disk, props.tile_desc);
    write_dataset(group, "tile_desc_offset_ptr", csr5.tile_desc_offset_ptr, H5T_NATIVE_UINT32, H5T_STD_U32LE,
                  props.tile_desc_offset_ptr);
    write_dataset(group, "tile_desc_offset", csr5.tile_desc_offset, H5T_NATIVE_UINT32, H5T_STD_U32LE,
                  props.tile_desc_offset);
}

auto load_csr5(group_view_t group) -> dim::mat::csr5<double> {
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
      .tile_desc
      = read_dataset<decltype(retval_t::tile_desc)>(group, "tile_desc", tile_types.in_memory, tile_types.on_disk),
      .tile_desc_offset_ptr = read_dataset<decltype(retval_t::tile_desc_offset_ptr)>(group, "tile_desc_offset_ptr",
                                                                                     H5T_NATIVE_UINT32, H5T_STD_U32LE),
      .tile_desc_offset = read_dataset<decltype(retval_t::tile_desc_offset)>(group, "tile_desc_offset",
                                                                             H5T_NATIVE_UINT32, H5T_STD_U32LE)};

    retval.tile_count = retval.tile_desc.size();
    retval.dimensions = {.rows = static_cast<uint32_t>(retval.row_ptr.size() - 1), .cols = num_cols};

    return retval;
}

auto read_vector(group_view_t group, const std::string& dataset_name) -> std::vector<double> {
    return detail::read_dataset<std::vector<double>>(group, dataset_name, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE);
}

auto is_hdf5(const std::filesystem::path& path) -> bool {
    constexpr std::byte magic[8] = {std::byte{0x89}, std::byte{'H'},  std::byte{'D'},  std::byte{'F'},
                                    std::byte{'\r'}, std::byte{'\n'}, std::byte{0x1a}, std::byte{'\n'}};

    auto file = io::open(path, "rb");

    std::byte buffer[8];

    return std::fread(std::data(buffer), sizeof(std::byte), std::size(buffer), file.get()) == std::size(buffer)
           && std::equal(std::begin(buffer), std::end(buffer), std::begin(magic));
}

using csr5_t = mat::csr5<double>;

template<>
struct type_translator_t<csr5_t::tile_descriptor_type>
{
    using underlying_t = csr5_t::tile_col_storage;
    using underlying_translator_t = type_translator_t<underlying_t>;

    static auto in_memory() noexcept -> type_t {
        return h5::type_t::create_array(underlying_translator_t::in_memory(), 4);
    }
    static auto on_disk() noexcept -> type_t { return h5::type_t::create_array(underlying_translator_t::on_disk(), 4); }
};

auto calculate_tile_chunk(dataspace_view_t dataspace, size_t part, size_t num_parts) {
    const auto total_tile_count = dataspace.get_dim();

    const auto part_size
      = static_cast<hsize_t>(std::ceil(static_cast<double>(total_tile_count) / static_cast<double>(num_parts)));

    const auto first_tile = part_size * part;
    const auto tile_count = std::min(part_size, total_tile_count - first_tile);

    return std::pair{first_tile, tile_count};
}

auto load_csr5_partial(group_view_t group, size_t part, size_t num_parts) -> mat::csr5<double> {
    constexpr auto tile_size = csr5_t::tile_size();

    const auto is_last_part = part == num_parts - 1;

    const auto tiles_dataset = group.open_dataset("tile_desc");
    const auto [first_tile, tile_count] = calculate_tile_chunk(tiles_dataset.get_dataspace(), part, num_parts);

    auto&& tile_desc = tiles_dataset.read_slab<decltype(csr5_t::tile_desc)>(first_tile, tile_count);

    auto&& tile_ptr = group.open_dataset("tile_ptr").read_slab<decltype(csr5_t::tile_ptr)>(first_tile, tile_count + 1);

    auto&& tile_desc_offset_ptr = group.open_dataset("tile_desc_offset_ptr")
                                    .read_slab<decltype(csr5_t::tile_desc_offset_ptr)>(first_tile, tile_count + 1);

    const auto tile_desc_offset_ptr_start = tile_desc_offset_ptr.front();
    const auto tile_desc_offset_ptr_end = tile_desc_offset_ptr.back();
    auto&& tile_desc_offset = group.open_dataset("tile_desc_offset")
                                .read_slab<decltype(csr5_t::tile_desc_offset)>(
                                  tile_desc_offset_ptr_start, tile_desc_offset_ptr_end - tile_desc_offset_ptr_start);

    const auto row_ptr_dataset = group.open_dataset("row_ptr");
    const auto row_ptr_length = row_ptr_dataset.get_dataspace().get_dim();
    const auto dimensions
      = mat::dimensions_t{.rows = static_cast<uint32_t>(row_ptr_dataset.get_dataspace().get_dim() - 1),
                          .cols = group.open_attribute("column_count").read<uint32_t>()};

    const auto row_start = tile_ptr.front();
    // for last process, we need to load even rows after the last tile to compute the tail sum.
    // +1 because indices are 0 based and the tile_ptr needs to access row_ptr[row_end]
    const auto row_end = is_last_part ? row_ptr_length : tile_ptr.back() + 1;
    auto&& row_ptr = group.open_dataset("row_ptr").read_slab<decltype(csr5_t::row_ptr)>(row_start, row_end - row_start);

    const auto vals_dataset = group.open_dataset("vals");
    const auto val_start = first_tile * tile_size;
    const auto val_end = is_last_part ? vals_dataset.get_dataspace().get_dim() : (val_start + tile_count * tile_size);

    const auto col_idx_dataset = group.open_dataset("col_idx");
    return csr5_t{.dimensions = dimensions,
                  .vals = vals_dataset.read_slab<decltype(csr5_t::vals)>(val_start, val_end - val_start),
                  .col_idx = col_idx_dataset.read_slab<decltype(csr5_t::col_idx)>(val_start, val_end - val_start),
                  .row_ptr = std::move(row_ptr),
                  .tile_count = tile_count,
                  .tile_ptr = std::move(tile_ptr),
                  .tile_desc = std::move(tile_desc),
                  .tile_desc_offset_ptr = std::move(tile_desc_offset_ptr),
                  .tile_desc_offset = std::move(tile_desc_offset),
                  // TODO: these might be more appropriate as arguments to the SpMV call.
                  .val_offset = val_start,
                  .skip_tail = !is_last_part};
}
} // namespace dim::io::h5