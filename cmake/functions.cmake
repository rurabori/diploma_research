# globs sources in a given directory and appends them to the list of sources.
macro(glob_dir directory)
  file(
    GLOB
    __tmp
    CONFIGURE_DEPENDS
    ${directory}/*.c
    ${directory}/*.cpp
    ${directory}/*.cu
    ${directory}/*.cuh
    ${directory}/*.h)
  list(APPEND sources ${__tmp})
endmacro()

# globs all sources in the directory. Plus directory `win32` on Windows,
# directory `unix` on unix-like systems, directory `linux` on linux and
# directory `darwin` on macos.
macro(glob_sources)
  # common sources.
  file(
    GLOB_RECURSE
    sources
    CONFIGURE_DEPENDS
    *.c
    *.cpp
    *.h
    *.cu
    *.cuh)

  if(WIN32)
    # windows specific stuff.
    glob_dir(win32)
  else()
    # all the stuff common for unix like systems.
    glob_dir(unix)

    if(APPLE)
      # apple specific stuff.
      glob_dir(darwin)
    else()
      # linux specific stuff.
      glob_dir(linux)
    endif()
  endif()
endmacro()

# adds an executable target, which takes all .cpp and .h files in its directory
# as its input (for OS specific check how `glob_sources` behaves). Initializes
# the settings to project defaults and adds versioning.
function(brr_add_executable target_name)
  glob_sources()

  add_executable(${target_name} ${sources})

  brr_target_init(${target_name})

  add_versioning(${target_name})

  target_include_directories(${target_name} PRIVATE .)
endfunction()

# adds a library target of specified type, which takes all .cpp and .h files in
# its directory (for OS specific check how `glob_sources` behaves) as well as .h
# files in include/<target_name> directory of the project as its input.
# Initializes the settings to project defaults and adds the include dir of the
# project as public include directory.
function(brr_add_library target_name type)

  if("${type}" STREQUAL "INTERFACE")
    add_library(${target_name} INTERFACE)

    brr_interface_target_init(${target_name})

    glob_dir(${PROJECT_SOURCE_DIR}/include/${target_name})

    target_include_directories(${target_name}
                               INTERFACE ${PROJECT_SOURCE_DIR}/include)
  else()
    glob_sources()
    glob_dir(${PROJECT_SOURCE_DIR}/include/${target_name})

    add_library(${target_name} ${type} ${sources})

    brr_target_init(${target_name})

    target_include_directories(
      ${target_name}
      PUBLIC ${PROJECT_SOURCE_DIR}/include
      PRIVATE .)
  endif()
endfunction()

add_custom_target(
  graphviz
  "${CMAKE_COMMAND}" "--graphviz=dim" .
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")


macro(conditional_add_subdirectory directory condition)
  if (${condition})
    add_subdirectory(${directory})
  endif()
endmacro()
