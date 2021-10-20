#include "dim/mat/storage_formats/csr5.h"
#include <doctest/doctest.h>

#include <algorithm>

#include <dim/io/matrix_market.h>
#include <dim/mat/storage_formats.h>
#include <immintrin.h>

using csr5_t = dim::mat::csr5<double>;
using tile_desc_t = csr5_t::tile_descriptor_type;
using tile_col_desc_t = tile_desc_t::tile_column_descriptor;

TEST_CASE("Test tile_col num_bits_set") {
    constexpr auto col = tile_col_desc_t{.y_offset = 4, .scansum_offset = 0, .bit_flag = 0b1000'1000'1000'1000};
    static_assert(col.num_bits_set(false) == 4,
                  "number of bits in columns other than first must be number of set bits.");
    static_assert(col.num_bits_set(true) == 5, "first column has first bit set implicitly.");

    constexpr auto col2 = tile_col_desc_t{.y_offset = 4, .scansum_offset = 0, .bit_flag = 0b1000'1000'1000'1001};
    static_assert(col2.num_bits_set(true) == 5,
                  "first column has first bit set implicitly but doesn't add to count if one is already there.");
}

TEST_CASE("Test conversion from CSR") {
    auto csr = dim::io::matrix_market::load_as_csr<double>("tests/test_files/mm.mtx");

    const auto csr5 = dim::mat::csr5<>::from_csr(csr);

    REQUIRE_EQ(csr5.dimensions.cols, csr.dimensions.cols);
    REQUIRE_EQ(csr5.dimensions.rows, csr.dimensions.rows);

    REQUIRE_FALSE(std::equal(csr5.vals.begin(), csr5.vals.end(), csr.values.begin()));

    REQUIRE_EQ(csr5.tile_desc[0],
               tile_desc_t{
                 .columns = {tile_col_desc_t{.y_offset = 0, .scansum_offset = 0, .bit_flag = 0b1000'0100'0010'0001},
                             tile_col_desc_t{.y_offset = 4, .scansum_offset = 0, .bit_flag = 0b1000'1000'1000'1000},
                             tile_col_desc_t{.y_offset = 8, .scansum_offset = 0, .bit_flag = 0b1010'1000'1000'1000},
                             tile_col_desc_t{.y_offset = 13, .scansum_offset = 0, .bit_flag = 0b1010'1010'1010'1010}}});

    REQUIRE_EQ(
      csr5.tile_desc[1],
      tile_desc_t{.columns = {tile_col_desc_t{.y_offset = 0, .scansum_offset = 3, .bit_flag = 0b0000'0000'0000'0010},
                              tile_col_desc_t{.y_offset = 2, .scansum_offset = 0, .bit_flag = 0b0000'0000'0000'0000},
                              tile_col_desc_t{.y_offset = 2, .scansum_offset = 0, .bit_flag = 0b0000'0000'0000'0000},
                              tile_col_desc_t{.y_offset = 2, .scansum_offset = 0, .bit_flag = 0b0000'0000'0000'0000}}});

    //csr5.spmv({}, {});

    // TODO: tests for tile_desc_offset when understood better.
}

auto vec_equal(__m128i vec, __m128i vec2) noexcept -> bool {
    return _mm_movemask_epi8(_mm_cmpeq_epi32(vec, vec2)) == 0xffff;
}

auto vec_equal(__m256i vec, __m256i vec2) noexcept -> bool {
    return _mm256_movemask_epi8(_mm256_cmpeq_epi32(vec, vec2)) == (int)0xffffffff;
}

TEST_CASE("Test tile descriptor vectorization") {
    constexpr auto descriptor = tile_desc_t{
      .columns = {tile_col_desc_t{.y_offset = 0, .scansum_offset = 3, .bit_flag = 0b1000'0100'0010'0001},
                  tile_col_desc_t{.y_offset = 4, .scansum_offset = 2, .bit_flag = 0b1000'1000'1000'1000},
                  tile_col_desc_t{.y_offset = 8, .scansum_offset = 1, .bit_flag = 0b1010'1000'1000'1000},
                  tile_col_desc_t{.y_offset = 13, .scansum_offset = 0, .bit_flag = 0b1010'1010'1010'1010}}};

    const auto desc_equal = [&descriptor](const auto& vec, auto&& accessor) {
        return vec_equal(vec, _mm_set_epi32(accessor(descriptor.columns[3]), accessor(descriptor.columns[2]),
                                            accessor(descriptor.columns[1]), accessor(descriptor.columns[0])));
    };

    const auto vec = descriptor.vectorized();

    REQUIRE(desc_equal(vec.y_offset, [](auto desc) { return desc.y_offset; }));
    REQUIRE(desc_equal(vec.scansum_offset, [](auto desc) { return desc.scansum_offset; }));
    REQUIRE(desc_equal(vec.bit_flag, [](auto desc) { return desc.bit_flag; }));

    REQUIRE(vec_equal(vec.get_local_bit(0), _mm256_set_epi64x(0, 0, 0, 1)));
    REQUIRE(vec_equal(vec.get_local_bit(1), _mm256_set_epi64x(1, 0, 0, 0)));
    REQUIRE(vec_equal(vec.get_local_bit(2), _mm256_set_epi64x(0, 0, 0, 0)));
    REQUIRE(vec_equal(vec.get_local_bit(3), _mm256_set_epi64x(1, 1, 1, 0)));
}
