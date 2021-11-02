#include <H5Ppublic.h>
#include <doctest/doctest.h>

#include <dim/io/h5.h>
#include <dim/mat/storage_formats.h>

#include <filesystem>

namespace h5 = dim::io::h5;

auto create_sample_matrix() {
    auto retval = dim::mat::coo<double>{dim::mat::dimensions_t{4, 4}, 4};
    const auto write_element = [&retval](size_t idx, double value) {
        retval.values[idx] = value;
        retval.col_indices[idx] = static_cast<uint32_t>(idx);
        retval.row_indices[idx] = static_cast<uint32_t>(idx);
    };

    write_element(0, 1.);
    write_element(1, 2.);
    write_element(2, 3.);
    write_element(3, 4.);

    return dim::mat::csr<double>::from_coo(retval);
}

TEST_CASE("test matlab/petsc compatible HDF5 roundtrip") {
    const std::string group_name{"/testing"};
    const auto sample = create_sample_matrix();
    const auto path = std::filesystem::temp_directory_path() / "test.hdf5";

    {
        auto file = h5::file_t::create(path, H5F_ACC_TRUNC);
        auto props = h5::plist_t::create(H5P_GROUP_CREATE);
        h5_try H5Pset_local_heap_size_hint(props.get_id(), 3);

        auto group = file.create_group(group_name, h5::plist_t::defaulted(), props, h5::plist_t::defaulted());
        h5::write_matlab_compatible(group.get_id(), sample);
    } // namespace dim::io::h5;

    // check the file was actually created.
    REQUIRE(std::filesystem::exists(path));

    auto file = h5::file_t::open(path, H5F_ACC_RDONLY);
    auto matrix_group = file.open_group(group_name);

    const auto result = h5::read_matlab_compatible(matrix_group.get_id());

    REQUIRE(std::equal(result.values.begin(), result.values.begin(), sample.values.begin()));
    REQUIRE(std::equal(result.col_indices.begin(), result.col_indices.begin(), sample.col_indices.begin()));
    REQUIRE(
      std::equal(result.row_start_offsets.begin(), result.row_start_offsets.begin(), sample.row_start_offsets.begin()));
    REQUIRE_EQ(result.dimensions.cols, sample.dimensions.cols);
    REQUIRE_EQ(result.dimensions.rows, sample.dimensions.rows);
}

auto range_equal(const auto& l, const auto& r) {
    return l.size() == r.size() && std::equal(l.begin(), l.end(), r.begin());
}

TEST_CASE("dim::io::csr5_hdf5_roundtrip") {
    const auto example = dim::mat::csr5<double>{
      .vals = {1., 2., 3., 4., 5.},
      .col_idx = {1, 2, 3, 4, 5},
      .row_ptr = {1, 2, 3, 4, 5},
      .tile_ptr = {1, 2, 3},
      .tile_desc = {{.columns = {{.y_offset = 0, .scansum_offset = 0, .bit_flag = 0b1000'0100'0010'0001},
                                 {.y_offset = 4, .scansum_offset = 0, .bit_flag = 0b1000'1000'1000'1000},
                                 {.y_offset = 8, .scansum_offset = 0, .bit_flag = 0b1010'1000'1000'1000},
                                 {.y_offset = 13, .scansum_offset = 0, .bit_flag = 0b1010'1010'1010'1010}}},
                    {.columns = {{.y_offset = 0, .scansum_offset = 3, .bit_flag = 0b0000'0000'0000'0010},
                                 {.y_offset = 2, .scansum_offset = 0, .bit_flag = 0b0000'0000'0000'0000},
                                 {.y_offset = 2, .scansum_offset = 0, .bit_flag = 0b0000'0000'0000'0000},
                                 {.y_offset = 2, .scansum_offset = 0, .bit_flag = 0b0000'0000'0000'0000}}}},
      .tile_desc_offset_ptr = {1, 2, 3, 4, 5},
      .tile_desc_offset = {1, 2, 3, 4, 5}};

    const std::string group_name{"testing"};
    const auto path = std::filesystem::temp_directory_path() / "test.csr5.hdf5";

    {
        auto file = h5::file_t::create(path, H5F_ACC_TRUNC);
        auto group = file.create_group(group_name);
        h5::store(group.get_id(), example);
    }

    // check the file was actually created.
    REQUIRE(std::filesystem::exists(path));

    auto file = h5::file_t::open(path, H5F_ACC_RDONLY);
    auto matrix_group = file.open_group(group_name);
    const auto result = h5::load_csr5(matrix_group.get_id());

    REQUIRE(range_equal(example.vals, result.vals));
    REQUIRE(range_equal(example.col_idx, result.col_idx));
    REQUIRE(range_equal(example.row_ptr, result.row_ptr));
    REQUIRE(range_equal(example.tile_ptr, result.tile_ptr));
    REQUIRE(range_equal(example.tile_desc, result.tile_desc));
    REQUIRE(range_equal(example.tile_desc_offset_ptr, result.tile_desc_offset_ptr));
    REQUIRE(range_equal(example.tile_desc_offset, result.tile_desc_offset));
}
