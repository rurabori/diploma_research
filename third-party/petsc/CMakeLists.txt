if (${system_scientific_libs}) 
    find_package(PkgConfig)
    pkg_check_modules(PETSc REQUIRED IMPORTED_TARGET PETSc)
    add_library(dim_tp_petsc INTERFACE)
    add_library(dim::tp::petsc ALIAS dim_tp_petsc)
    target_link_libraries(dim_tp_petsc INTERFACE PkgConfig::PETSc)
else()
    find_package(petsc REQUIRED CONFIG)
    add_library(dim::tp::petsc ALIAS petsc::petsc)
endif()





