brr_add_executable(csr5_orig_benchmark)

target_link_libraries(csr5_orig_benchmark PRIVATE dim::simple_main dim::io
                                                  dim::bench csr5_orig)
