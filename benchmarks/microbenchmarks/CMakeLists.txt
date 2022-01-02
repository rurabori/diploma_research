find_package(benchmark REQUIRED)

brr_add_executable(microbenchmarks)
target_link_libraries(microbenchmarks PRIVATE benchmark::benchmark dim::io
                                              dim::mat csr5_orig)
