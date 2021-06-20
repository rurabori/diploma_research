#pragma once

#include <cuda_interop/annotations.h>
#include <driver_types.h>

namespace cui {

struct error_handler
{
    const char* file{};
    const char* function{};
    const size_t line{};

    __host__ void operator+(cudaError error) const;
};

} // namespace cui

#define cuda_try cui::error_handler{__FILE__, __FUNCTION__, __LINE__} +
