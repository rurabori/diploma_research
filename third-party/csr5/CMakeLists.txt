project(csr5)
find_package(OpenMP REQUIRED COMPONENTS CXX)

brr_add_library(csr5 INTERFACE)
target_link_libraries(csr5 INTERFACE OpenMP::OpenMP_CXX dim::memory)
