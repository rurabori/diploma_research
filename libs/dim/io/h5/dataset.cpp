#include <dim/io/h5/dataset.h>

#include <dim/io/h5/err.h>

namespace dim::io::h5 {

auto dataset_view_t::write(const void* data, type_view_t type, dataspace_view_t mem_space, dataspace_view_t file_space,
                           plist_view_t props) -> void {
    h5_try ::H5Dwrite(get_id(), type.get_id(), mem_space.get_id(), file_space.get_id(), props.get_id(), data);
}

auto dataset_view_t::get_type() const noexcept -> type_t { return type_t{::H5Dget_type(get_id())}; }

auto dataset_view_t::get_dataspace() const -> dataspace_t { return dataspace_t{::H5Dget_space(get_id())}; }
auto dataset_view_t::read(void* data, type_view_t type, dataspace_view_t mem_space, dataspace_view_t file_space,
                          plist_view_t props) -> void {
    h5_try ::H5Dread(get_id(), type.get_id(), mem_space.get_id(), file_space.get_id(), props.get_id(), data);
}

} // namespace dim::io::h5
