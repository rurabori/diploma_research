brr_add_library(dim_mat STATIC)
add_library(dim::mat ALIAS dim_mat)
target_link_libraries(dim_mat PUBLIC dim::memory)
