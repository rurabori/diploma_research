brr_add_executable(cg_dim)

target_link_libraries(cg_dim PRIVATE dim::simple_main dim::mpi dim::io
                                     dim::csr5_mpi)

install(TARGETS cg_dim RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
