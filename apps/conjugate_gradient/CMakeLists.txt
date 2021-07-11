find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(fmt REQUIRED)
find_package(tclap REQUIRED)
find_package(magic_enum REQUIRED)
find_package(MPI REQUIRED)

add_executable(conjugate_gradient main.cpp matrix_storage_formats.h
                                  spmv_algos.hpp timed_section.h)
brr_target_init(conjugate_gradient)
add_versioning(conjugate_gradient)

target_link_libraries(
  conjugate_gradient
  PRIVATE mmio
          csr5
          OpenMP::OpenMP_CXX
          fmt::fmt
          tclap::tclap
          magic_enum::magic_enum
          dim::memory
          MPI::MPI_CXX)

if(${enable_cuda})
  target_sources(conjugate_gradient csr5_cuda.cu cuda_algo_facade)
  target_compile_definitions(conjugate_gradient PRIVATE CUDA_ENABLED)
  target_link_libraries(conjugate_gradient PRIVATE csr5_cuda)
endif()
