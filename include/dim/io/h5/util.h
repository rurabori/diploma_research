#ifndef INCLUDE_DIM_IO_H5_UTIL
#define INCLUDE_DIM_IO_H5_UTIL
#include <H5Ipublic.h>
#include <hdf5.h>
#include <stdexcept>
#include <utility>

namespace dim::io::h5 {

template<const auto& Fun, typename... Args>
auto handle_create(Args&&... args) -> hid_t {
    auto handle = Fun(std::forward<Args>(args)...);
    if (!::H5Iis_valid(handle))
        throw std::runtime_error{"HDF5 handle invalid"};

    return handle;
}

} // namespace dim::io::h5

#endif /* INCLUDE_DIM_IO_H5_UTIL */
