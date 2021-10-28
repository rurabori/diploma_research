#ifndef INCLUDE_DIM_IO_H5_OBJECT
#define INCLUDE_DIM_IO_H5_OBJECT

#include <dim/io/h5/fwd.h>
#include <dim/io/h5/location.h>

namespace dim::io::h5 {

struct object_view_t : public location_view_t
{
    using location_view_t::location_view_t;

    [[nodiscard]] auto create_attribute(const std::string& name, type_view_t type, dataspace_view_t dataspace,
                                        plist_view_t creation = plist_t::defaulted(),
                                        plist_view_t access = plist_t::defaulted()) const -> attribute_t;

    [[nodiscard]] auto open_attribute(const std::string& name, plist_view_t access = plist_t::defaulted()) const
      -> attribute_t;
};

} // namespace dim::io::h5

#endif /* INCLUDE_DIM_IO_H5_OBJECT */
