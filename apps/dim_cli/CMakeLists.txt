find_package(fmt REQUIRED)
find_package(structopt REQUIRED)
find_package(spdlog REQUIRED)
find_package(magic_enum REQUIRED)
find_package(CURL REQUIRED)
find_package(libarchive REQUIRED CONFIG)
find_package(scn REQUIRED CONFIG)
find_package(yaml-cpp REQUIRED CONFIG)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED CONFIG)

brr_add_executable(dim_cli)

target_link_libraries(
  dim_cli
  PRIVATE fmt::fmt
          dim::io
          dim::simple_main
          structopt::structopt
          spdlog::spdlog
          magic_enum::magic_enum
          CURL::libcurl
          libarchive::libarchive
          scn::scn
          Threads::Threads
          yaml-cpp::yaml-cpp
          opencv::opencv)

install(TARGETS dim_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES ${CMAKE_HOME_DIRECTORY}/resources/config.yaml TYPE DATA)
