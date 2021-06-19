#include <cuda_interop/error.h>

#include <stdexcept>

#include <fmt/format.h>
#include <cuda_runtime_api.h>

namespace cui {

void error_handler::operator+(cudaError error) const {
    if (error == cudaSuccess)
        return;

    throw std::runtime_error{fmt::format("Cuda call at {}({}):{} failed with error code {}: {}.", file, function, line,
                                         error, cudaGetErrorString(error))};
}

} // namespace cui