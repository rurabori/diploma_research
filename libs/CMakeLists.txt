add_subdirectory(mmio)
add_subdirectory(memory_ext)

if(${enable_cuda_backend})
  add_subdirectory(cuda_interop)
endif()
