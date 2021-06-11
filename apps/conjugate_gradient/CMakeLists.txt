find_package(OpenMP REQUIRED)

brr_add_executable(conjugate_gradient)
target_link_libraries(conjugate_gradient PRIVATE mmio csr5 OpenMP::OpenMP_CXX)
