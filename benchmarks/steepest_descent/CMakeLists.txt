brr_add_executable(steepest_descent)

target_link_libraries(steepest_descent PRIVATE dim::simple_main dim::io)

install(TARGETS steepest_descent RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
