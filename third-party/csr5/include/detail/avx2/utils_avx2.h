#ifndef UTILS_AVX2_H
#define UTILS_AVX2_H

#include "common_avx2.h"

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

// sum up 4 double-precision numbers
inline double hsum_avx(__m256d in256d) {
    double sum;

    __m256d hsum = _mm256_add_pd(in256d, _mm256_permute2f128_pd(in256d, in256d, 0x1));
    _mm_store_sd(&sum, _mm_hadd_pd(_mm256_castpd256_pd128(hsum), _mm256_castpd256_pd128(hsum)));

    return sum;
}

// sum up 8 single-precision numbers
inline float hsum_avx(__m256 in256) {
    float sum;

    __m256 hsum = _mm256_hadd_ps(in256, in256);
    hsum = _mm256_add_ps(hsum, _mm256_permute2f128_ps(hsum, hsum, 0x1));
    _mm_store_ss(&sum, _mm_hadd_ps(_mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum)));

    return sum;
}

// exclusive scan using a single thread
template<typename T>
void scan_single(T* s_scan, const int l) {
    T old_val, new_val;

    old_val = s_scan[0];
    s_scan[0] = 0;
    for (int i = 1; i < l; i++) {
        new_val = s_scan[i];
        s_scan[i] = old_val + s_scan[i - 1];
        old_val = new_val;
    }
}

// inclusive prefix-sum scan
inline __m256d hscan_avx(__m256d in256d) {
    __m256d t0, t1;
    t0 = _mm256_permute4x64_pd(in256d, 0x93);
    t1 = _mm256_add_pd(in256d, _mm256_blend_pd(t0, _mm256_set1_pd(0), 0x1));

    t0 = _mm256_permute4x64_pd(in256d, 0x4E);
    t1 = _mm256_add_pd(t1, _mm256_blend_pd(t0, _mm256_set1_pd(0), 0x3));

    t0 = _mm256_permute4x64_pd(in256d, 0x39);
    t1 = _mm256_add_pd(t1, _mm256_blend_pd(t0, _mm256_set1_pd(0), 0x7));

    return t1;
}

// inclusive prefix-sum scan
inline __m256 hscan_avx(__m256 in256) {
    __m256 t0, t1;
    // shift1_AVX + add
    t0 = _mm256_permute_ps(in256, _MM_SHUFFLE(2, 1, 0, 3));
    t1 = _mm256_permute2f128_ps(t0, t0, 41);
    in256 = _mm256_add_ps(in256, _mm256_blend_ps(t0, t1, 0x11));
    // shift2_AVX + add
    t0 = _mm256_permute_ps(in256, _MM_SHUFFLE(1, 0, 3, 2));
    t1 = _mm256_permute2f128_ps(t0, t0, 41);
    in256 = _mm256_add_ps(in256, _mm256_blend_ps(t0, t1, 0x33));
    // shift3_AVX + add
    in256 = _mm256_add_ps(in256, _mm256_permute2f128_ps(in256, in256, 41));
    return in256;
}

} // namespace csr5::avx2
#endif // UTILS_AVX2_H
