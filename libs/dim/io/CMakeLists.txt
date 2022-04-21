find_package(scnlib REQUIRED)
find_package(fmt REQUIRED)

brr_add_library(dim_io STATIC)
add_library(dim::io ALIAS dim_io)

target_link_libraries(dim_io PUBLIC mmio dim::mat scnlib::scnlib dim::tp::hdf5 fmt::fmt dim::mpi)
