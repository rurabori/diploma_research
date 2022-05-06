find_package(nlohmann_json REQUIRED)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  find_package(TBB REQUIRED COMPONENTS tbb)
endif()

brr_add_executable(cg_dim)

target_link_libraries(cg_dim PRIVATE dim::simple_main dim::mpi dim::io
                                     nlohmann_json::nlohmann_json)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_libraries(cg_dim PRIVATE TBB::tbb)
endif()
