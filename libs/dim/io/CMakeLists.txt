find_package(scn REQUIRED)
find_package(HDF5 REQUIRED CONFIG)
find_package(fmt REQUIRED)

brr_add_library(dim_io STATIC)
add_library(dim::io ALIAS dim_io)

target_link_libraries(dim_io PUBLIC mmio dim::mat scn::scn HDF5::HDF5 HDF5:CXX fmt::fmt)
