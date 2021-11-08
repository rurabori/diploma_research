add_subdirectory(csr5)
add_subdirectory(hdf5)
add_subdirectory(petsc)

conditional_add_subdirectory(csr5_cuda ${enable_cuda_backend})
