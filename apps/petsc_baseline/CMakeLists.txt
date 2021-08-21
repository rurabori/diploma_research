find_package(PkgConfig REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(HDF5 REQUIRED CONFIG)

pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)

brr_add_executable(petsc_baseline)
target_link_libraries(petsc_baseline PRIVATE dim::mpi fmt::fmt PkgConfig::PETSC
                                             spdlog::spdlog HDF5::HDF5)

                                             
install(TARGETS petsc_baseline RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
