find_package(fmt REQUIRED)
find_package(HDF5 REQUIRED)

brr_add_executable(hdf5_converter)
target_link_libraries(hdf5_converter PRIVATE fmt::fmt HDF5::HDF5)
