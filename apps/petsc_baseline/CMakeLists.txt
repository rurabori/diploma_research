find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(HDF5 REQUIRED CONFIG)
find_package(petsc REQUIRED CONFIG)

brr_add_executable(petsc_baseline)
target_link_libraries(petsc_baseline PRIVATE dim::mpi fmt::fmt petsc::petsc
                                             spdlog::spdlog HDF5::HDF5)

                                             
install(TARGETS petsc_baseline RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
