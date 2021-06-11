add_library(csr5 INTERFACE include/anonymouslib_avx512.h)
target_include_directories(csr5 INTERFACE "${CMAKE_CURRENT_LIST_DIR}/include")
