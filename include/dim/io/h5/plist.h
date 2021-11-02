#ifndef INCLUDE_DIM_IO_H5_PLIST
#define INCLUDE_DIM_IO_H5_PLIST

#include <dim/io/h5/view.h>

#include <H5Ppublic.h>

namespace dim::io::h5 {

struct plist_view_t : public view_t
{
    using view_t::view_t;
    static constexpr plist_view_t defaulted() noexcept { return plist_view_t{H5P_DEFAULT}; }
};

class plist_t : public view_wrapper_t<plist_view_t, H5Pclose>
{
    using super_t = view_wrapper_t<plist_view_t, H5Pclose>;
    using super_t::super_t;

public:
    static plist_t create(hid_t plist_class);
};
} // namespace dim::io::h5
#endif /* INCLUDE_DIM_IO_H5_PLIST */
