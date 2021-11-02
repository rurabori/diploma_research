find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(fmt REQUIRED)
find_package(tclap REQUIRED)
find_package(magic_enum REQUIRED)
find_package(scn REQUIRED)

add_executable(csr5_bench main.cpp spmv_algos.hpp timed_section.h)
brr_target_init(csr5_bench)
add_versioning(csr5_bench)

target_link_libraries(
  csr5_bench
  PRIVATE mmio
          csr5
          OpenMP::OpenMP_CXX
          fmt::fmt
          tclap::tclap
          magic_enum::magic_enum
          dim::memory
          dim::mat
          dim::io
          dim::mpi
          scn::scn)

if(${enable_cuda})
  target_sources(csr5_bench csr5_cuda.cu cuda_algo_facade)
  target_compile_definitions(csr5_bench PRIVATE CUDA_ENABLED)
  target_link_libraries(csr5_bench PRIVATE csr5_cuda)
endif()
