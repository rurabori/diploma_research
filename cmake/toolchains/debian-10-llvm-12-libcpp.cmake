set(tools /usr/bin)
set(CMAKE_C_COMPILER ${tools}/clang-12)
set(CMAKE_CXX_COMPILER ${tools}/clang++-12)

set(CMAKE_CXX_FLAGS "-stdlib=libc++")
