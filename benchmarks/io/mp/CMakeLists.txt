find_package(nlohmann_json REQUIRED)

brr_add_executable(dim_bench_io_mp)

target_link_libraries(dim_bench_io_mp PRIVATE dim::simple_main dim::mpi dim::io
                                     dim::csr5_mpi nlohmann_json::nlohmann_json)

install(TARGETS dim_bench_io_mp RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
