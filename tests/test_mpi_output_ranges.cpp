#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

#include <dim/mpi/csr5/output_range.h>

using dim::mpi::csr5::output_range_t;

TEST_CASE("single runs") {
    std::vector<output_range_t> runs{{0, 1}, {2, 4}, {5, 6}};

    REQUIRE_EQ(output_range_t::syncs_downto(runs, 0), 0);
    REQUIRE_EQ(output_range_t::syncs_downto(runs, 1), 1);
    REQUIRE_EQ(output_range_t::syncs_downto(runs, 2), 2);
}

TEST_CASE("overlapping runs") {
    std::vector<output_range_t> runs{{0, 1}, {1, 4}, {4, 6}};

    REQUIRE_EQ(output_range_t::syncs_downto(runs, 0), 0);
    REQUIRE_EQ(output_range_t::syncs_downto(runs, 1), 0);
    REQUIRE_EQ(output_range_t::syncs_downto(runs, 2), 1);
}

TEST_CASE("contiguous overlapping runs") {
    std::vector<output_range_t> runs{{0, 1}, {1, 1}, {1, 1}, {1, 4}, {4, 6}};

    REQUIRE_EQ(output_range_t::syncs_downto(runs, 0), 0);
    REQUIRE_EQ(output_range_t::syncs_downto(runs, 1), 0);
    REQUIRE_EQ(output_range_t::syncs_downto(runs, 2), 0);
    REQUIRE_EQ(output_range_t::syncs_downto(runs, 3), 0);
    REQUIRE_EQ(output_range_t::syncs_downto(runs, 4), 3);
}
