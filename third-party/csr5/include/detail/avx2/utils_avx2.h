#ifndef THIRD_PARTY_CSR5_INCLUDE_DETAIL_AVX2_UTILS_AVX2
#define THIRD_PARTY_CSR5_INCLUDE_DETAIL_AVX2_UTILS_AVX2

#include "common_avx2.h"
#include <array>

namespace csr5::avx2 {

template<typename iT>
size_t binary_search_right_boundary_kernel(const iT* row_pointer, const iT key_input, const size_t size) {
    size_t start = 0;
    size_t stop = size - 1;
    iT key_median;

    while (stop >= start) {
        auto median = (stop + start) / 2;

        key_median = row_pointer[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start;
}

// TODO: this relies on permutation argument being imm8, research if that is correct.
template<std::integral... Ty>
consteval int make_permute_seq(Ty... seq) {
    int value{};
    for (size_t off = 0; auto pos : std::array{seq...})
        value |= pos << (off++ * 8 / sizeof...(seq));

    return value;
}

// sum up 4 double-precision numbers
inline double hsum_avx(__m256d in256d) {
    __m256d hsum = _mm256_add_pd(in256d, _mm256_permute2f128_pd(in256d, in256d, make_permute_seq(1, 0)));

    // NOLINTNEXTLINE - initialization would be dead write.
    double sum;
    _mm_store_sd(&sum, _mm_hadd_pd(_mm256_castpd256_pd128(hsum), _mm256_castpd256_pd128(hsum)));

    return sum;
}

// sum up 8 single-precision numbers
inline float hsum_avx(__m256 in256) {
    __m256 hsum = _mm256_hadd_ps(in256, in256);
    hsum = _mm256_add_ps(hsum, _mm256_permute2f128_ps(hsum, hsum, make_permute_seq(1, 0)));

    // NOLINTNEXTLINE - initialization would be dead write.
    float sum;
    _mm_store_ss(&sum, _mm_hadd_ps(_mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum)));

    return sum;
}

// inclusive prefix-sum scan
inline __m256d hscan_avx(__m256d in256d) {
    auto t0 = _mm256_permute4x64_pd(in256d, make_permute_seq(3, 0, 1, 2));
    auto t1 = _mm256_add_pd(in256d, _mm256_blend_pd(t0, _mm256_set1_pd(0), 0b0001));

    t0 = _mm256_permute4x64_pd(in256d, make_permute_seq(2, 3, 0, 1));
    t1 = _mm256_add_pd(t1, _mm256_blend_pd(t0, _mm256_set1_pd(0), 0b0011));

    t0 = _mm256_permute4x64_pd(in256d, make_permute_seq(1, 2, 3, 0));
    t1 = _mm256_add_pd(t1, _mm256_blend_pd(t0, _mm256_set1_pd(0), 0b0111));

    return t1;
}

// inclusive prefix-sum scan
inline __m256 hscan_avx(__m256 in256) {
    constexpr auto zero = 0b1000;

    // shift1_AVX + add
    auto t0 = _mm256_permute_ps(in256, make_permute_seq(3, 0, 1, 2));
    auto t1 = _mm256_permute2f128_ps(t0, t0, make_permute_seq(zero, 0));
    in256 = _mm256_add_ps(in256, _mm256_blend_ps(t0, t1, 0b00010001));

    // shift2_AVX + add
    t0 = _mm256_permute_ps(in256, make_permute_seq(2, 3, 0, 1));
    t1 = _mm256_permute2f128_ps(t0, t0, make_permute_seq(zero, 0));
    in256 = _mm256_add_ps(in256, _mm256_blend_ps(t0, t1, 0b00110011));

    // shift3_AVX + add
    in256 = _mm256_add_ps(in256, _mm256_permute2f128_ps(in256, in256, make_permute_seq(zero, 0)));

    return in256;
}

} // namespace csr5::avx2
#endif /* THIRD_PARTY_CSR5_INCLUDE_DETAIL_AVX2_UTILS_AVX2 */
