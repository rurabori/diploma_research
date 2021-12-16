add_subdirectory(dim)
conditional_add_subdirectory(mkl ${enable_mkl_benchmark})
conditional_add_subdirectory(petsc ${enable_petsc_benchmark})
