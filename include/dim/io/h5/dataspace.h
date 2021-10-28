#ifndef INCLUDE_DIM_IO_H5_DATASPACE
#define INCLUDE_DIM_IO_H5_DATASPACE

#include <dim/io/h5/view.h>
#include <dim/span.h>

#include <H5Spublic.h>

#include <stdexcept>

namespace dim::io::h5 {

struct dataspace_view_t : public view_t
{
    static constexpr auto all() -> dataspace_view_t { return dataspace_view_t{H5S_ALL}; }

    [[nodiscard]] auto get_ndims() const noexcept -> size_t;

    auto get_dims(dim::span<hsize_t> dims) const -> void;

    [[nodiscard]] auto get_dim() const -> hsize_t;
};

class dataspace_t : public view_wrapper_t<dataspace_view_t, H5Sclose>
{
    using super_t = view_wrapper_t<dataspace_view_t, H5Sclose>;
    using super_t::super_t;

public:
    static auto create(H5S_class_t cls) -> dataspace_t;
    static auto create(std::span<const hsize_t> dims) -> dataspace_t;
};

} // namespace dim::io::h5

#endif /* INCLUDE_DIM_IO_H5_DATASPACE */
