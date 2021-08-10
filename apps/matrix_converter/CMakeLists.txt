find_package(fmt REQUIRED)
find_package(HDF5 REQUIRED CONFIG)
find_package(tclap REQUIRED)
find_package(spdlog REQUIRED)
find_package(magic_enum REQUIRED)

brr_add_executable(matrix_converter)
target_link_libraries(
  matrix_converter PRIVATE fmt::fmt HDF5::HDF5 dim::io tclap::tclap
                           spdlog::spdlog magic_enum::magic_enum)
