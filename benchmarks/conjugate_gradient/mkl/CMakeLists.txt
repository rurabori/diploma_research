set(MKL_MPI "openmpi")
set(MKL_TARGET_ARCH "intel64")


find_package(MKL)

if(NOT ${MKL_FOUND})
  message(STATUS "MKL not found, disabling gradient descent benchmark.")
  return()
endif()

find_package(magic_enum REQUIRED)

brr_add_executable(cg_mkl)

target_link_libraries(cg_mkl PRIVATE MKL::MKL dim::io dim::simple_main
                                     magic_enum::magic_enum)

install(TARGETS cg_mkl RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
