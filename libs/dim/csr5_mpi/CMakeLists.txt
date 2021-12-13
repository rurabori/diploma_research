brr_add_library(dim_csr5_mpi STATIC)

add_library(dim::csr5_mpi ALIAS dim_csr5_mpi)

target_link_libraries(dim_csr5_mpi PUBLIC dim::mpi dim::mat dim::io)
