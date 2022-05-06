find_package(nlohmann_json REQUIRED)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  find_package(TBB REQUIRED COMPONENTS tbb)
endif()

brr_add_executable(dim_sp)

target_link_libraries(dim_sp PRIVATE dim::simple_main dim::io dim::mat
                                     nlohmann_json::nlohmann_json)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_libraries(dim_sp PRIVATE TBB::tbb)
endif()
