#include <dim/io/h5/object.h>

#include <dim/io/h5/attribute.h>

#include <H5Apublic.h>

namespace dim::io::h5 {

auto object_view_t::create_attribute(const std::string& name, type_view_t type, dataspace_view_t dataspace,
                                     plist_view_t creation, plist_view_t access) const -> attribute_t {
    return attribute_t{
      ::H5Acreate(get_id(), name.c_str(), type.get_id(), dataspace.get_id(), creation.get_id(), access.get_id())};
}

auto object_view_t::open_attribute(const std::string& name, plist_view_t access) const -> attribute_t {
    return attribute_t{::H5Aopen(get_id(), name.c_str(), access.get_id())};
}
} // namespace dim::io::h5
