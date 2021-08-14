find_package(fmt REQUIRED)
find_package(HDF5 REQUIRED CONFIG)
find_package(structopt REQUIRED)
find_package(spdlog REQUIRED)
find_package(magic_enum REQUIRED)

brr_add_executable(dim_cli)
target_link_libraries(
  dim_cli PRIVATE fmt::fmt HDF5::HDF5 dim::io structopt::structopt
                  spdlog::spdlog magic_enum::magic_enum)
