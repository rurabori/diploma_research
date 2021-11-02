#ifndef INCLUDE_DIM_IO_H5_LOCATION
#define INCLUDE_DIM_IO_H5_LOCATION

#include <dim/io/h5/dataspace.h>
#include <dim/io/h5/fwd.h>
#include <dim/io/h5/plist.h>
#include <dim/io/h5/type.h>
#include <dim/io/h5/view.h>

#include <string>

namespace dim::io::h5 {

struct location_view_t : public view_t
{
    using view_t::view_t;

    [[nodiscard]] auto create_group(const std::string& name, plist_view_t link_creation = plist_t::defaulted(),
                                    plist_view_t group_creation = plist_t::defaulted(),
                                    plist_view_t group_access = plist_t::defaulted()) const -> group_t;

    [[nodiscard]] auto open_group(const std::string& name, plist_view_t group_access = plist_t::defaulted()) const
      -> group_t;

    [[nodiscard]] auto create_dataset(const std::string& name, type_view_t type, dataspace_view_t dataspace,
                                      plist_view_t link_creation = plist_t::defaulted(),
                                      plist_view_t data_creation = plist_t::defaulted(),
                                      plist_view_t data_access = plist_t::defaulted()) const -> dataset_t;

    [[nodiscard]] auto open_dataset(const std::string& name, plist_view_t data_access = plist_t::defaulted()) const
      -> dataset_t;
};

} // namespace dim::io::h5

#endif /* INCLUDE_DIM_IO_H5_LOCATION */
