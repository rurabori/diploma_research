#ifndef INCLUDE_DIM_IO_H5_TYPE
#define INCLUDE_DIM_IO_H5_TYPE

#include <dim/io/h5/fwd.h>
#include <dim/io/h5/view.h>
#include <dim/span.h>

#include <H5Tpublic.h>

namespace dim::io::h5 {

struct type_view_t : public view_t
{
    using view_t::view_t;

    auto operator==(type_view_t o) const noexcept { return ::H5Tequal(get_id(), o.get_id()) > 0; }
};

class type_t : public view_wrapper_t<type_view_t, H5Tclose>
{
    using super_t = view_wrapper_t<type_view_t, H5Tclose>;
    using super_t::super_t;

    friend object_view_t;

public:
    static auto create_array(type_view_t base, dim::span<const hsize_t> dims) -> type_t {
        return type_t{::H5Tarray_create(base.get_id(), static_cast<unsigned int>(dims.size()), dims.data())};
    }

    static auto create_array(type_view_t base, hsize_t dim) -> type_t {
        return create_array(base, dim::span<const hsize_t>{&dim, 1});
    }
};
} // namespace dim::io::h5

#endif /* INCLUDE_DIM_IO_H5_TYPE */
