#ifndef INCLUDE_DIM_IO_H5_ERR
#define INCLUDE_DIM_IO_H5_ERR

#include <H5public.h>
#include <stdexcept>

namespace dim::io::h5 {
struct h5_try_
{
    void operator%(herr_t err) const {
        // TODO: better error message.
        if (err < 0)
            throw std::runtime_error{"HDF5 failed."};
    }
};

} // namespace dim::io::h5

// NOLINTNEXTLINE
#define h5_try dim::io::h5::h5_try_{} %

#endif /* INCLUDE_DIM_IO_H5_ERR */
