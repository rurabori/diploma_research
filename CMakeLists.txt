cmake_minimum_required(VERSION 3.15)

project(
  dim
  VERSION 1.0.0
  LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# setup options from commandline.
include(options)

# default project settings.
include(default_settings)

# versioning functions.
include(versioning)

# helper functions.
include(functions)

# conan integration.
include(conan)

# code coverage
include(coverage)

# default installation directories for OSes.
include(GNUInstallDirs)

add_subdirectory(libs)
add_subdirectory(apps)

# third-party dependencies
add_subdirectory(third-party)

include(CTest)
conditional_add_subdirectory(tests ${BUILD_TESTING})

add_subdirectory(benchmarks)

include(CPack)
