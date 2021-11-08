find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

brr_add_executable(petsc_baseline)
target_link_libraries(petsc_baseline PRIVATE dim::mpi fmt::fmt dim::tp::petsc
                                             spdlog::spdlog dim::tp::hdf5)

                                             
install(TARGETS petsc_baseline RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})