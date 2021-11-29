brr_add_executable(mpi_csr5)
target_link_libraries(mpi_csr5 dim::simple_main dim::mpi dim::io dim::mat)
install(TARGETS mpi_csr5 RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

