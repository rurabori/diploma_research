set(MKL_MPI "openmpi")

find_package(MKL)

if(NOT ${MKL_FOUND})
  message(STATUS "MKL not found, disabling gradient descent benchmark.")
  return()
endif()

find_package(magic_enum REQUIRED)

brr_add_executable(gradient_descent_mkl)

target_link_libraries(
  gradient_descent_mkl PRIVATE MKL::MKL dim::io dim::simple_main
                               magic_enum::magic_enum)

install(TARGETS gradient_descent_mkl
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
