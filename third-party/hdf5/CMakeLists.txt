add_library(dim_tp_hdf5 INTERFACE)
add_library(dim::tp::hdf5 ALIAS dim_tp_hdf5)

if(${system_scientific_libs})
  find_package(PkgConfig)
  pkg_check_modules(HDF5 REQUIRED IMPORTED_TARGET hdf5)

  target_link_libraries(dim_tp_hdf5 INTERFACE PkgConfig::HDF5)
else()
  find_package(hdf5 REQUIRED CONFIG)
  target_link_libraries(dim_tp_hdf5 INTERFACE hdf5::hdf5)
endif()
