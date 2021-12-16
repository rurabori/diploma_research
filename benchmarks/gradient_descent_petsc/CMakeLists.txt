find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

brr_add_executable(gradient_descent_petsc)
target_link_libraries(
  gradient_descent_petsc PRIVATE dim::mpi fmt::fmt dim::tp::petsc
                                 spdlog::spdlog dim::tp::hdf5)

target_compile_definitions(gradient_descent_petsc
                           PRIVATE HDF5_DISABLE_VERSION_CHECK=1)

install(TARGETS gradient_descent_petsc
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
