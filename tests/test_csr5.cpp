#include <doctest/doctest.h>

#include <algorithm>

#include <dim/io/matrix_market.h>
#include <dim/mat/storage_formats.h>

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
}
