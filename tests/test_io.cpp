#include <H5Fpublic.h>
#include <bits/ranges_algobase.h>
#include <doctest/doctest.h>

#include <dim/io/h5.h>
#include <dim/mat/storage_formats.h>

#include <hdf5/H5File.h>

#include <filesystem>

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
    const std::string group_name{"testing"};
    const auto sample = create_sample_matrix();
    const auto path = std::filesystem::temp_directory_path() / "test.hdf5";

    {
        H5::H5File file{path, H5F_ACC_TRUNC};
        auto matrix_group = file.createGroup(group_name, 3);
        dim::io::h5::write_matlab_compatible(matrix_group, sample);
    }

    // check the file was actually created.
    REQUIRE(std::filesystem::exists(path));

    H5::H5File file{path, H5F_ACC_RDONLY};
    auto matrix_group = file.openGroup(group_name);
    const auto result = dim::io::h5::read_matlab_compatible(matrix_group);

    REQUIRE(std::equal(result.values.begin(), result.values.begin(), sample.values.begin()));
    REQUIRE(std::equal(result.col_indices.begin(), result.col_indices.begin(), sample.col_indices.begin()));
    REQUIRE(
      std::equal(result.row_start_offsets.begin(), result.row_start_offsets.begin(), sample.row_start_offsets.begin()));
    REQUIRE_EQ(result.dimensions.cols, sample.dimensions.cols);
    REQUIRE_EQ(result.dimensions.rows, sample.dimensions.rows);
}