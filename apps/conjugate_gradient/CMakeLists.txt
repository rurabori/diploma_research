find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(fmt REQUIRED)
find_package(tclap REQUIRED)

brr_add_executable(conjugate_gradient)
target_link_libraries(
  conjugate_gradient PRIVATE mmio csr5 OpenMP::OpenMP_CXX fmt::fmt tclap::tclap
                             csr5_cuda)
