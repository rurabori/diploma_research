find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)

brr_add_executable(cg_petsc)
target_link_libraries(
  cg_petsc
  PRIVATE dim::mpi
          fmt::fmt
          dim::tp::petsc
          spdlog::spdlog
          dim::tp::hdf5
          dim::io
          dim::bench
          nlohmann_json::nlohmann_json)

target_compile_definitions(cg_petsc PRIVATE HDF5_DISABLE_VERSION_CHECK=1)
