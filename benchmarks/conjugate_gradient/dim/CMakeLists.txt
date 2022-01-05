find_package(nlohmann_json REQUIRED)

brr_add_executable(cg_dim)

target_link_libraries(cg_dim PRIVATE dim::simple_main dim::mpi dim::io
                                     nlohmann_json::nlohmann_json)
