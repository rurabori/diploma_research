find_package(doctest REQUIRED)

brr_add_executable(unit_tests)
target_link_libraries(unit_tests PRIVATE doctest::doctest dim::io dim::mat
                                         dim::csr5_mpi)

doctest_discover_tests(unit_tests WORKING_DIRECTORY ${CMAKE_HOME_DIRECTORY}
                       ADD_LABELS 1)
add_coverage(unit_tests)
