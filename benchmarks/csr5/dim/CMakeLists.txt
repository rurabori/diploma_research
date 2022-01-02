find_package(fmt REQUIRED)
find_package(structopt REQUIRED)
find_package(magic_enum REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)

brr_add_executable(csr5_single_node)

target_link_libraries(
  csr5_single_node
  PRIVATE structopt::structopt magic_enum::magic_enum dim::io spdlog::spdlog
          nlohmann_json::nlohmann_json dim::bench)
