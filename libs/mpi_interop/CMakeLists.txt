find_package(MPI REQUIRED COMPONENTS CXX)

brr_add_library(dim_mpi STATIC)
add_library(dim::mpi ALIAS dim_mpi)

target_link_libraries(dim_mpi PUBLIC MPI::MPI_CXX)
