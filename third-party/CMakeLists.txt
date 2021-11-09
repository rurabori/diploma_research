add_subdirectory(csr5)
add_subdirectory(hdf5)

conditional_add_subdirectory(petsc ${enable_petsc_benchmark})
conditional_add_subdirectory(csr5_cuda ${enable_cuda_backend})
