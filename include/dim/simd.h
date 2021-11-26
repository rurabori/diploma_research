#ifndef INCLUDE_DIM_SIMD
#define INCLUDE_DIM_SIMD

#include <array>
#include <concepts>

#include <dim/bit.h>

#include <immintrin.h>

namespace dim::simd {

/**
 * @brief creates a permutation for avx permute calls.
 *
 * @param seq sequence of numbers to create a permutation from.
 * @return int a value suitable to send to avx permute calls.
 */
consteval auto make_permute_seq(std::integral auto... seq) -> int {
    int value{};
    size_t off = 0;
    for (auto pos : std::array{seq...})
        value |= pos << (off++ * 8 / sizeof...(seq));

    return value;
}

inline auto hscan_avx(__m256d in256d) -> __m256d {
    auto t0 = _mm256_permute4x64_pd(in256d, make_permute_seq(3, 0, 1, 2));
    auto t1 = _mm256_add_pd(in256d, _mm256_blend_pd(t0, _mm256_set1_pd(0), 0b0001));

    t0 = _mm256_permute4x64_pd(in256d, make_permute_seq(2, 3, 0, 1));
    t1 = _mm256_add_pd(t1, _mm256_blend_pd(t0, _mm256_set1_pd(0), 0b0011));

    t0 = _mm256_permute4x64_pd(in256d, make_permute_seq(1, 2, 3, 0));
    t1 = _mm256_add_pd(t1, _mm256_blend_pd(t0, _mm256_set1_pd(0), 0b0111));

    return t1;
}

inline auto hsum_avx(__m256d in256d) -> double {
    __m256d hsum = _mm256_add_pd(in256d, _mm256_permute2f128_pd(in256d, in256d, make_permute_seq(1, 0)));

    // NOLINTNEXTLINE - initialization would be dead write.
    double sum;
    _mm_store_sd(&sum, _mm_hadd_pd(_mm256_castpd256_pd128(hsum), _mm256_castpd256_pd128(hsum)));

    return sum;
}
/**
 * @brief vec[i] = vec[i + desc[i]]
 *
 * @param vec to shuffle.
 * @param desc relative offsets for each element in vec.
 * @return __m256d shuffled vec.
 */
inline auto shuffle_relative(__m256d vec, __m128i desc) noexcept -> __m256d {
    // add base relative offsets, cast to 256 bits.
    auto shuffle_mask = _mm256_castsi128_si256(_mm_add_epi32(desc, _mm_set_epi32(3, 2, 1, 0)));
    // copy each 32bit part from the lower 128 bits to its neighbour.
    shuffle_mask = _mm256_permutevar8x32_epi32(shuffle_mask, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
    // x2 as we're shuffling 32-bit halves but aliasing 64-bit doubles (e.g 1st double is at index 2 etc.).
    shuffle_mask = _mm256_add_epi32(shuffle_mask, shuffle_mask);
    // +1 to every second to have "upper" and "lower" parts of the doubles.
    shuffle_mask = _mm256_add_epi32(shuffle_mask, _mm256_set_epi32(1, 0, 1, 0, 1, 0, 1, 0));
    // shuffle by the created mask.
    return _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(vec), shuffle_mask));
}

inline auto any_bit_set(__m256i vec) -> bool {
    // _mm256_testz_si256 returns 1 if vec & 0xFF == 0, hence if it returns 0, some bit was set.
    return !_mm256_testz_si256(vec, _mm256_set1_epi64x(all_bits_set<int64_t>));
}

/**
 * @brief Merges 2 vectors by a mask, bits where [mask] is 0 are taken from [vec0] bits where [mask] is 1 are taken from
 * [vec1].
 *
 * @param vec0 vec to take bits from if mask bit is 0.
 * @param vec1 vec to take bits from if mask bit is 1.
 * @param mask mask for the merge.
 * @return __m256d merged vector.
 */
inline auto merge_vec(__m256d vec0, __m256d vec1, __m256i mask) noexcept -> __m256d {
    return _mm256_or_pd(_mm256_andnot_pd(_mm256_castsi256_pd(mask), vec0),
                        _mm256_and_pd(_mm256_castsi256_pd(mask), vec1));
}

} // namespace dim::simd

#endif /* INCLUDE_DIM_SIMD */
