#ifndef COMMON_AVX2_H
#define COMMON_AVX2_H

#include <stdint.h>

#include "immintrin.h"
#include <omp.h>

#include "../common.h"
#include "../utils.h"

namespace csr5::avx2 {

constexpr auto ANONYMOUSLIB_CSR5_OMEGA = 4;
constexpr auto ANONYMOUSLIB_CSR5_SIGMA = 16;
constexpr auto ANONYMOUSLIB_X86_CACHELINE = 64;

} // namespace csr5::avx2

#endif // COMMON_AVX2_H
