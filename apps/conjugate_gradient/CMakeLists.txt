find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(fmt REQUIRED)
find_package(fire-hpp REQUIRED)

brr_add_executable(conjugate_gradient)
target_link_libraries(conjugate_gradient PRIVATE mmio csr5 OpenMP::OpenMP_CXX
                                                 fmt::fmt fire-hpp::fire-hpp)
