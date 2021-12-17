find_package(fmt REQUIRED)
find_package(MPI REQUIRED COMPONENTS C CXX)

brr_add_library(dim_mpi STATIC)
add_library(dim::mpi ALIAS dim_mpi)

target_link_libraries(dim_mpi PUBLIC MPI::MPI_C MPI::MPI_CXX fmt::fmt dim::mat
                                     dim::io)
