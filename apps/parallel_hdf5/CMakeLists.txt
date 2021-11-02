find_package(fmt REQUIRED CONFIG)
brr_add_executable(parallel_hdf5)

target_link_libraries(parallel_hdf5 PRIVATE dim::mpi dim::io fmt::fmt)

install(TARGETS parallel_hdf5 RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

