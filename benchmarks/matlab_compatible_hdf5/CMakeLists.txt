find_package(structopt REQUIRED)
find_package(spdlog REQUIRED)

brr_add_executable(matlab_compatible_hdf5)

target_link_libraries(matlab_compatible_hdf5
                      PRIVATE dim::io structopt::structopt spdlog::spdlog)

install(TARGETS matlab_compatible_hdf5
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
