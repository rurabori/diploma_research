#include "dim/mat/storage_formats/csr5.h"
#include <doctest/doctest.h>

#include <dim/bit.h>
#include <dim/simd.h>
#include <immintrin.h>

TEST_CASE("simd::any_bit_set") {
    REQUIRE(dim::simd::any_bit_set(_mm256_set_epi64x(0, 0, 0, 1)));
    REQUIRE_FALSE(dim::simd::any_bit_set(_mm256_set_epi64x(0, 0, 0, 0)));
}

auto vec_equal(__m256d vec, __m256d vec2) noexcept -> bool {
    return _mm256_movemask_epi8(_mm256_cmpeq_epi64(vec, vec2)) == dim::all_bits_set<int>;
}

TEST_CASE("simd::shuffle") {
    const auto base_vec = _mm256_set_pd(4., 3., 2., 1.);

    // don't move at all.
    REQUIRE(vec_equal(dim::simd::shuffle_relative(base_vec, _mm_set_epi32(0, 0, 0, 0)), base_vec));
    REQUIRE(vec_equal(dim::simd::shuffle_relative(base_vec, _mm_set_epi32(0, 1, 1, 1)), _mm256_set_pd(4., 4., 3., 2.)));
    REQUIRE(vec_equal(dim::simd::shuffle_relative(base_vec, _mm_set_epi32(0, 1, 2, 3)), _mm256_set_pd(4., 4., 4., 4.)));
}

TEST_CASE("simd::hscan_avx") {
    const auto base_vec = _mm256_set_pd(4., 3., 2., 1.);

    REQUIRE(vec_equal(dim::simd::hscan_avx(base_vec), _mm256_set_pd(10., 6., 3., 1.)));
}
