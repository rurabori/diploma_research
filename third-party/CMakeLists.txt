add_subdirectory(csr5)

if(${enable_cuda_backend})
  add_subdirectory(csr5_cuda)
endif()
