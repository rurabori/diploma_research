if (${system_scientific_libs}) 
    find_package(PkgConfig)
    pkg_check_modules(HDF5 REQUIRED IMPORTED_TARGET hdf5)
    
    add_library(dim_tp_hdf5 INTERFACE)
    add_library(dim::tp::hdf5 ALIAS dim_tp_hdf5)

    target_link_libraries(dim_tp_hdf5 INTERFACE PkgConfig::HDF5)
else()
    find_package(HDF5 REQUIRED CONFIG)
    add_library(dim::tp::hdf5 ALIAS HDF5::HDF5)
endif()





