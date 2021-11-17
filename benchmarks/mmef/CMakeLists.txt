find_package(structopt REQUIRED)
find_package(spdlog REQUIRED)

brr_add_executable(mmef)

target_link_libraries(mmef PRIVATE dim::io structopt::structopt spdlog::spdlog)
install(TARGETS mmef RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
