find_package(scn REQUIRED)

brr_add_library(dim_io STATIC)
add_library(dim::io ALIAS dim_io)

target_link_libraries(dim_io PUBLIC mmio dim::mat scn::scn)
