#include <H5Fpublic.h>
#include <H5Ppublic.h>
#include <doctest/doctest.h>

#include <dim/io/h5.h>
#include <dim/mat/storage_formats.h>

#include <concepts>
#include <filesystem>
#include <fstream>

namespace h5 = dim::io::h5;

TEST_CASE("h5::plist conversion from H5P_DEFAULT") {
    constexpr auto def = h5::plist_t::defaulted();
    static_assert(def.get_id() == H5P_DEFAULT);
    static_assert(std::convertible_to<h5::plist_t, h5::plist_view_t>);
}

TEST_CASE("h5::is_hdf5") {
    SUBCASE("should be h5") {
        const auto path = std::filesystem::temp_directory_path() / "test.is_hdf5.hdf5";
        (void)h5::file_t::create(path, H5F_ACC_TRUNC);
        REQUIRE(h5::is_hdf5(path));
    }

    SUBCASE("should not be h5") {
        const auto path = std::filesystem::temp_directory_path() / "test.is_not_hdf5.hdf5";
        std::ofstream{path} << "this is certainly not a HDF5 magic";
        REQUIRE_FALSE(h5::is_hdf5(path));
    }
}
