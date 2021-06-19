#pragma once

#include <driver_types.h>

namespace cui {

struct error_handler
{
    const char* file{};
    const char* function{};
    const size_t line{};

    void operator+(cudaError error) const;
};

} // namespace cui

#define cuda_try cui::error_handler{__FILE__, __FUNCTION__, __LINE__} +
