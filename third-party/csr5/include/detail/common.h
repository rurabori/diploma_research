#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

namespace csr5::avx2 {

using namespace std;

constexpr auto ANONYMOUSLIB_SUCCESS = 0;
constexpr auto ANONYMOUSLIB_UNKOWN_FORMAT = -1;
constexpr auto ANONYMOUSLIB_UNSUPPORTED_CSR5_OMEGA = -2;
constexpr auto ANONYMOUSLIB_CSR_TO_CSR5_FAILED = -3;
constexpr auto ANONYMOUSLIB_UNSUPPORTED_CSR_SPMV = -4;
constexpr auto ANONYMOUSLIB_UNSUPPORTED_VALUE_TYPE = -5;

constexpr auto ANONYMOUSLIB_FORMAT_CSR = 0;
constexpr auto ANONYMOUSLIB_FORMAT_CSR5 = 1;
constexpr auto ANONYMOUSLIB_FORMAT_HYB5 = 2;

} // namespace csr5::avx2

#endif // COMMON_H
