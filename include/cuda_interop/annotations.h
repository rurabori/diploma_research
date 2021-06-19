#pragma once

#include <cuda_runtime_api.h>

#define CUI_API __host__ __device__

#ifdef __CUDA_ARCH__
#define fast_sqrt(x) (1. / rsqrt((x)))
#else
#define fast_sqrt(x) std::sqrt((x))
#endif
