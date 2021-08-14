add_subdirectory(mmio)
add_subdirectory(mpi_interop)
add_subdirectory(dim)

if(${enable_cuda_backend})
  add_subdirectory(cuda_interop)
endif()
