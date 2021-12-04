#add_subdirectory(csr5_bench)
conditional_add_subdirectory(petsc_baseline ${enable_petsc_benchmark})
add_subdirectory(dim_cli)
