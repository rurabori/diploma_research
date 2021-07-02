find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(fmt REQUIRED)
find_package(tclap REQUIRED)
find_package(magic_enum REQUIRED)
find_package(MPI REQUIRED)

brr_add_executable(conjugate_gradient)
target_link_libraries(
  conjugate_gradient
  PRIVATE mmio
          csr5
          OpenMP::OpenMP_CXX
          fmt::fmt
          tclap::tclap
          csr5_cuda
          magic_enum::magic_enum
          cui::cui
          dim::memory
          MPI::MPI_CXX)
