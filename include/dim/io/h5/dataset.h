#ifndef INCLUDE_DIM_IO_H5_DATASET
#define INCLUDE_DIM_IO_H5_DATASET

#include <dim/io/h5/dataspace.h>
#include <dim/io/h5/location.h>
#include <dim/io/h5/plist.h>
#include <dim/io/h5/type.h>

#include <H5Dpublic.h>

namespace dim::io::h5 {

struct dataset_view_t : public view_t
{
    auto write(const void* data, type_view_t type, dataspace_view_t mem_space = dataspace_view_t::all(),
               dataspace_view_t file_space = dataspace_view_t::all(), plist_view_t props = plist_t::defaulted())
      -> void;

    auto read(void* data, type_view_t type, dataspace_view_t mem_space = dataspace_view_t::all(),
              dataspace_view_t file_space = dataspace_view_t::all(), plist_view_t props = plist_t::defaulted()) -> void;

    [[nodiscard]] auto get_dataspace() const -> dataspace_t;

    [[nodiscard]] auto get_type() const noexcept -> type_t;
};

class dataset_t : public view_wrapper_t<dataset_view_t, H5Dclose>
{
    using super_t = view_wrapper_t<dataset_view_t, H5Dclose>;
    using super_t::super_t;

    friend location_view_t;

public:
};

} // namespace dim::io::h5

#endif /* INCLUDE_DIM_IO_H5_DATASET */
