find_package(nlohmann_json REQUIRED)

brr_add_executable(dim_sp)

target_link_libraries(dim_sp PRIVATE dim::simple_main dim::io dim::mat
                                     nlohmann_json::nlohmann_json)
