find_package(fmt REQUIRED)

brr_add_executable(petsc_baseline)
target_link_libraries(petsc_baseline PRIVATE dim::mpi fmt::fmt)
