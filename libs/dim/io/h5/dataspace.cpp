#include <dim/io/h5/dataspace.h>

namespace dim::io::h5 {

auto dataspace_t::create(std::span<const hsize_t> dims) -> dataspace_t {
    return dataspace_t{::H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr)};
}

auto dataspace_t::create(H5S_class_t cls) -> dataspace_t { return dataspace_t{::H5Screate(cls)}; }

auto dataspace_view_t::get_ndims() const noexcept -> size_t {
    return static_cast<size_t>(::H5Sget_simple_extent_ndims(get_id()));
}
auto dataspace_view_t::get_dims(dim::span<hsize_t> dims) const -> void {
    if (dims.size() != get_ndims()) [[unlikely]]
        throw std::runtime_error{"dims could not be written to the provided span"};

    ::H5Sget_simple_extent_dims(get_id(), dims.data(), nullptr);
}
auto dataspace_view_t::get_dim() const -> hsize_t {
    hsize_t result{};
    get_dims({&result, 1});
    return result;
}
} // namespace dim::io::h5
