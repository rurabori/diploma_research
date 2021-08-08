find_package(fmt REQUIRED)
find_package(HDF5 REQUIRED)
find_package(tclap REQUIRED)

brr_add_executable(matrix_converter)
target_link_libraries(matrix_converter PRIVATE fmt::fmt HDF5::HDF5 dim::io
                                             tclap::tclap)
