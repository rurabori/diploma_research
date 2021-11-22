find_package(spdlog REQUIRED)
find_package(structopt REQUIRED)
find_package(fmt REQUIRED)

brr_add_library(dim_simple_main INTERFACE)
add_library(dim::simple_main ALIAS dim_simple_main)

target_link_libraries(dim_simple_main INTERFACE spdlog::spdlog
                                                structopt::structopt fmt::fmt)
