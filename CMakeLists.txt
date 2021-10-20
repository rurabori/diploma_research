cmake_minimum_required(VERSION 3.15)

# setup options from commandline.
include(cmake/options.cmake)

project(
  small_vector
  VERSION 1.0.0
  LANGUAGES C CXX)

if(${enable_cuda_backend})
  enable_language(CUDA)
endif()

# default project settings.
include(cmake/default_settings.cmake)

# versioning functions.
include(cmake/versioning.cmake)

# helper functions.
include(cmake/functions.cmake)

# conan integration.
include(cmake/conan.cmake)

# code coverage
include(cmake/coverage.cmake)

# default installation directories for OSes.
include(GNUInstallDirs)

add_subdirectory(libs)
add_subdirectory(apps)

# third-party dependencies
add_subdirectory(third-party)

include(CTest)
add_subdirectory(tests)
# add_subdirectory(benchmarks)

include(CPack)
