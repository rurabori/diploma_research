#ifndef INCLUDE_DIM_IO_H5_GROUP
#define INCLUDE_DIM_IO_H5_GROUP

#include <dim/io/h5/object.h>

#include <H5Gpublic.h>


namespace dim::io::h5 {
struct group_view_t : public object_view_t
{
    using object_view_t::object_view_t;
};

class group_t : public view_wrapper_t<group_view_t, H5Gclose>
{
    using super_t = view_wrapper_t<group_view_t, H5Gclose>;
    using super_t::super_t;

    friend location_view_t;

public:
};

} // namespace dim::io::h5

#endif /* INCLUDE_DIM_IO_H5_GROUP */
