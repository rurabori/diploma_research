find_package(fmt REQUIRED)
find_package(HDF5 REQUIRED CONFIG)
find_package(structopt REQUIRED)
find_package(spdlog REQUIRED)
find_package(magic_enum REQUIRED)
find_package(CURL REQUIRED)
find_package(LibArchive REQUIRED CONFIG)
find_package(scn REQUIRED CONFIG)
find_package(yaml-cpp REQUIRED CONFIG)

brr_add_executable(dim_cli)
target_link_libraries(
  dim_cli
  PRIVATE fmt::fmt
          HDF5::HDF5
          dim::io
          structopt::structopt
          spdlog::spdlog
          magic_enum::magic_enum
          CURL::CURL
          LibArchive::LibArchive
          scn::scn
          yaml-cpp::yaml-cpp)

install(TARGETS dim_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
