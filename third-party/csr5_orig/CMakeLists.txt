project(csr5_orig)
find_package(OpenMP REQUIRED COMPONENTS C)

add_library(csr5_orig INTERFACE)
target_include_directories(csr5_orig SYSTEM INTERFACE include)
target_link_libraries(csr5_orig INTERFACE OpenMP::OpenMP_C)
