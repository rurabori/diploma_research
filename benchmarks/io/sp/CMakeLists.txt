find_package(nlohmann_json REQUIRED)

brr_add_executable(dim_bench_io_sp)

target_link_libraries(dim_bench_io_sp PRIVATE dim::simple_main dim::io
                                     nlohmann_json::nlohmann_json)

install(TARGETS dim_bench_io_sp RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

