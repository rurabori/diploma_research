#include <dim/io/h5.h>
#include <dim/io/h5/dataset.h>
#include <dim/io/h5/dataspace.h>
#include <dim/io/h5/type.h>
#include <dim/mat/storage_formats/base.h>

namespace dim::io::h5 {

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

template<>
struct type_translator_t<csr5_t::tile_ptr_type>
{
    using underlying_t = csr5_t::tile_ptr_type::storage_t;
    using underlying_translator_t = type_translator_t<underlying_t>;

    static auto in_memory() noexcept -> decltype(auto) { return underlying_translator_t::in_memory(); }
    static auto on_disk() noexcept -> decltype(auto) { return underlying_translator_t::on_disk(); }
};

dataset_props_t::operator plist_t() const {
    auto result = plist_t::create(H5P_DATASET_CREATE);
    if (chunk_size)
        ::H5Pset_chunk(result.get_id(), 1, std::addressof(*chunk_size));

    if (compression_level)
        ::H5Pset_deflate(result.get_id(), *compression_level);

    return result;
}

auto store(group_view_t group, const dim::mat::csr5<double>& csr5, const csr5_storage_props_t& props) -> void {
    write_attribute_2(group, "column_count", csr5.dimensions.cols);

    write_dataset_2(group, "vals", csr5.vals, props.vals);
    write_dataset_2(group, "col_idx", csr5.col_idx, props.col_idx);
    write_dataset_2(group, "row_ptr", csr5.row_ptr, props.row_ptr);
    write_dataset_2(group, "tile_ptr", csr5.tile_ptr, props.tile_ptr);
    write_dataset_2(group, "tile_desc", csr5.tile_desc, props.tile_desc);
    write_dataset_2(group, "tile_desc_offset_ptr", csr5.tile_desc_offset_ptr, props.tile_desc_offset_ptr);
    write_dataset_2(group, "tile_desc_offset", csr5.tile_desc_offset, props.tile_desc_offset);
}

auto load_csr5(group_view_t group) -> dim::mat::csr5<double> {
    using retval_t = mat::csr5<double>;

    retval_t retval{.vals = group.open_dataset("vals"),
                    .col_idx = group.open_dataset("col_idx"),
                    .row_ptr = group.open_dataset("row_ptr"),
                    .tile_ptr = group.open_dataset("tile_ptr"),
                    .tile_desc = group.open_dataset("tile_desc"),
                    .tile_desc_offset_ptr = group.open_dataset("tile_desc_offset_ptr"),
                    .tile_desc_offset = group.open_dataset("tile_desc_offset")};

    retval.tile_count = retval.tile_desc.size();
    retval.dimensions = mat::dimensions_t{.rows = static_cast<uint32_t>(retval.row_ptr.size() - 1),
                                          .cols = group.open_attribute("column_count")};

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

auto calculate_tile_chunk(dataspace_view_t dataspace, csr5_partial_identifier_t part) {
    const auto total_tile_count = dataspace.get_dim();

    const auto part_size
      = static_cast<hsize_t>(std::ceil(static_cast<double>(total_tile_count) / static_cast<double>(part.total_count)));

    const auto first_tile = part_size * part.idx;
    const auto tile_count = std::min(part_size, total_tile_count - first_tile);

    return std::pair{first_tile, tile_count};
}

auto load_csr5_partial(group_view_t group, csr5_partial_identifier_t part) -> mat::csr5<double> {
    constexpr auto tile_size = csr5_t::tile_size();

    const auto is_last_part = part.idx == (part.total_count - 1);

    const auto tiles_dataset = group.open_dataset("tile_desc");
    const auto [first_tile, tile_count] = calculate_tile_chunk(tiles_dataset.get_dataspace(), part);

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

    const auto row_start = tile_ptr.front().idx();
    // for last process, we need to load even rows after the last tile to compute the tail sum.
    // +1 because indices are 0 based and the tile_ptr needs to access row_ptr[row_end]
    const auto row_end = is_last_part ? row_ptr_length : tile_ptr.back().idx() + 1;
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
auto load_csr5(const std::filesystem::path& path, const std::string& group_name) -> mat::csr5<double> {
    const auto in = h5::file_t::open(path, H5F_ACC_RDONLY);
    return dim::io::h5::load_csr5(in.open_group(group_name));
}
} // namespace dim::io::h5