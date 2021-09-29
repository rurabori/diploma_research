#include <doctest/doctest.h>

#include <dim/io/format.h>

using dim::io::formattable_bytes;

TEST_CASE("test bytes formatting") {
    REQUIRE_EQ(fmt::format("{}", formattable_bytes{700}), "700B");
    REQUIRE_EQ(fmt::format("{}", formattable_bytes{1024}), "1kB");
    REQUIRE_EQ(fmt::format("{}", formattable_bytes{1'048'576}), "1MB");
    REQUIRE_EQ(fmt::format("{}", formattable_bytes{1'073'741'824}), "1GB");
    REQUIRE_EQ(fmt::format("{}/s", formattable_bytes{500.}), "500B/s");
}