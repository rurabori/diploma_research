project(csr5_orig)
find_package(OpenMP REQUIRED COMPONENTS C)

brr_add_library(csr5_orig STATIC)
target_link_libraries(csr5_orig PUBLIC OpenMP::OpenMP_C)
