find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

brr_add_executable(cg_petsc)
target_link_libraries(cg_petsc PRIVATE dim::mpi fmt::fmt dim::tp::petsc
                                       spdlog::spdlog dim::tp::hdf5 dim::io)

target_compile_definitions(cg_petsc PRIVATE HDF5_DISABLE_VERSION_CHECK=1)

install(TARGETS cg_petsc RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
