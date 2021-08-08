find_package(fmt REQUIRED)
find_package(HDF5 REQUIRED)

brr_add_executable(dist_spmv)
target_link_libraries(dist_spmv PRIVATE dim::mpi fmt::fmt HDF5::HDF5)
