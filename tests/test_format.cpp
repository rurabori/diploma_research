#include <doctest/doctest.h>

#include <iostream>
#include <string>

#include <dim/io/format.h>

using dim::io::formattable_bytes;

TEST_CASE("test bytes formatting") {
    REQUIRE_EQ(fmt::format("{}", formattable_bytes{700}), "700B");
    REQUIRE_EQ(fmt::format("{}", formattable_bytes{1024}), "1kiB");
    REQUIRE_EQ(fmt::format("{}", formattable_bytes{1'048'576}), "1MiB");
    REQUIRE_EQ(fmt::format("{}", formattable_bytes{1'073'741'824}), "1GiB");
    REQUIRE_EQ(fmt::format("{}/s", formattable_bytes{500.}), "500B/s");
}