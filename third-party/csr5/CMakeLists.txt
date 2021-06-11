find_package(OpenMP REQUIRED)

add_library(csr5 INTERFACE include/anonymouslib_avx2.h)
target_include_directories(csr5 INTERFACE "${CMAKE_CURRENT_LIST_DIR}/include")
target_link_libraries(csr5 INTERFACE OpenMP::OpenMP_CXX)