set(tools /usr/bin)
set(CMAKE_C_COMPILER ${tools}/clang-12)
set(CMAKE_CXX_COMPILER ${tools}/clang++-12)

set(CMAKE_CXX_FLAGS "-stdlib=libc++")
list(APPEND CMAKE_PREFIX_PATH "/usr/lib/petsc/lib/pkgconfig/")
set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH ON)
