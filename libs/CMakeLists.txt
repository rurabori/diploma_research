add_subdirectory(mmio)
add_subdirectory(memory_ext)
add_subdirectory(mpi_interop)
add_subdirectory(io)
add_subdirectory(mat)

if(${enable_cuda_backend})
  add_subdirectory(cuda_interop)
endif()
