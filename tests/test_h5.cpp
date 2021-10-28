#include <H5Ppublic.h>
#include <doctest/doctest.h>

#include <dim/io/h5.h>
#include <dim/mat/storage_formats.h>

#include <concepts>
#include <filesystem>

namespace h5 = dim::io::h5;

TEST_CASE("h5::plist conversion from H5P_DEFAULT") {
    constexpr auto def = h5::plist_t::defaulted();
    static_assert(def.get_id() == H5P_DEFAULT);
    static_assert(std::convertible_to<h5::plist_t, h5::plist_view_t>);
}