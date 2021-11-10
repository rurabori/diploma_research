find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(fmt REQUIRED)
find_package(structopt REQUIRED)
find_package(magic_enum REQUIRED)
find_package(scn REQUIRED)

brr_add_executable(csr5_single_node)

target_link_libraries(
  csr5_single_node
  PRIVATE mmio
          csr5
          OpenMP::OpenMP_CXX
          fmt::fmt
          structopt::structopt
          magic_enum::magic_enum
          dim::memory
          dim::mat
          dim::io
          dim::mpi
          scn::scn)
