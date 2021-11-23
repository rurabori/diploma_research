brr_add_executable(csr5_orig_benchmark)

target_link_libraries(csr5_orig_benchmark PRIVATE dim::simple_main dim::io
                                                  csr5_orig)
install(TARGETS csr5_orig_benchmark RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
