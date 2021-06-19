find_package(CUDAToolkit REQUIRED)

add_library(csr5_cuda INTERFACE include/anonymouslib_cuda.cuh)

# include as system for now to suppress warnings, fix in future. the samples dir
# is there because CSR5 uses headers from it, should remove in the future.
target_include_directories(
  csr5_cuda SYSTEM INTERFACE "${CMAKE_CURRENT_LIST_DIR}/include"
                             "${CUDAToolkit_LIBRARY_ROOT}/samples/common/inc")
target_link_libraries(csr5 INTERFACE CUDA::cudart)
