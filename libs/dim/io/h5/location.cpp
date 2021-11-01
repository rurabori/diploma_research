#include <dim/io/h5/location.h>

#include <dim/io/h5/dataset.h>
#include <dim/io/h5/group.h>
#include <dim/io/h5/util.h>

#include <H5Gpublic.h>

namespace dim::io::h5 {

auto location_view_t::create_group(const std::string& name, plist_view_t link_creation, plist_view_t group_creation,
                                   plist_view_t group_access) const -> group_t {
    return group_t{handle_create<::H5Gcreate>(get_id(), name.c_str(), link_creation.get_id(), group_creation.get_id(),
                                              group_access.get_id())};
}

auto location_view_t::open_group(const std::string& name, plist_view_t group_access) const -> group_t {
    return group_t{handle_create<::H5Gopen>(get_id(), name.c_str(), group_access.get_id())};
}

auto location_view_t::create_dataset(const std::string& name, type_view_t type, dataspace_view_t dataspace,
                                     plist_view_t link_creation, plist_view_t data_creation,
                                     plist_view_t data_access) const -> dataset_t {
    return dataset_t{handle_create<::H5Dcreate>(get_id(), name.c_str(), type.get_id(), dataspace.get_id(),
                                                link_creation.get_id(), data_creation.get_id(), data_access.get_id())};
}

auto location_view_t::open_dataset(const std::string& name, plist_view_t data_access) const -> dataset_t {
    return dataset_t{handle_create<::H5Dopen>(get_id(), name.c_str(), data_access.get_id())};
}

} // namespace dim::io::h5
