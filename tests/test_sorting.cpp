#include <doctest/doctest.h>

#include <random>
#include <ranges>

#include <dim/keyval_iterator.h>

void sort_and_check(auto&& a, auto&& b) {
    dim::keyval_sort(a, b);

    REQUIRE(std::ranges::is_sorted(a));
    REQUIRE(std::ranges::is_sorted(b));
}

TEST_CASE("dim::keyval_sort") {
    std::vector<int> a{2, 1, 4, 3, 5};
    std::vector<std::string> b{"2", "1", "4", "3", "5"};

    sort_and_check(a, b);
}

TEST_CASE("dim::keyval_sort long sequence") {
    std::mt19937_64 rng{};
    std::uniform_int_distribution<int> dist{};

    std::vector<int> a;
    std::generate_n(std::back_inserter(a), 500, [&] { return dist(rng); });

    std::vector<int> b{a};

    sort_and_check(a, b);
}