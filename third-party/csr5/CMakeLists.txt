project(csr5)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(fmt REQUIRED)

brr_add_library(csr5 STATIC)
target_link_libraries(csr5 PUBLIC OpenMP::OpenMP_CXX dim::memory fmt::fmt)
