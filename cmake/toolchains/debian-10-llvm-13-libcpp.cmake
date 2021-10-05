set(tools /usr/bin)
set(CMAKE_C_COMPILER ${tools}/clang-13)
set(CMAKE_CXX_COMPILER ${tools}/clang++-13)

set(CMAKE_CXX_FLAGS "-stdlib=libc++")
