find_package(fmt REQUIRED)
find_package(structopt REQUIRED)
find_package(magic_enum REQUIRED)
find_package(spdlog REQUIRED)

brr_add_executable(csr5_single_node)

target_link_libraries(
  csr5_single_node
  PRIVATE mmio
          csr5
          fmt::fmt
          structopt::structopt
          magic_enum::magic_enum
          dim::memory
          dim::mat
          dim::io
          spdlog::spdlog)
