#include <dim/io/h5/attribute.h>

namespace dim::io::h5 {

auto attribute_view_t::get_type() const noexcept -> type_t { return type_t{::H5Aget_type(get_id())}; }

auto attribute_view_t::write(const void* data, type_view_t type) -> void {
    h5_try ::H5Awrite(get_id(), type.get_id(), data);
}
auto attribute_view_t::read(void* data, type_view_t type) const -> void {
    h5_try ::H5Aread(get_id(), type.get_id(), data);
}
} // namespace dim::io::h5
