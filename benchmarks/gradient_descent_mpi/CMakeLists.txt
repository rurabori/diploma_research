brr_add_executable(gradient_descent_mpi)

target_link_libraries(gradient_descent_mpi PRIVATE dim::simple_main dim::mpi
                                                   dim::io dim::csr5_mpi)

install(TARGETS gradient_descent_mpi
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
