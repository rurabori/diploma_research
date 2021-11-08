# generated by conan install . -if build but not required for the build.

include(${CMAKE_CURRENT_LIST_DIR}/conan_integration.cmake)

set(__conan_config_files_dir ${CMAKE_BINARY_DIR}/__conan_config_files)

macro(ternary boolean value1 value2)
    if (${${boolean}})
      set(__conan_${boolean} ${value1})
    else()
      set(__conan_${boolean} ${value2})
    endif()
endmacro()

if(${invoke_conan} AND NOT EXISTS "${__conan_config_files_dir}")
  set(__conan_configurations_to_build ${CMAKE_BUILD_TYPE})
  if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    list(APPEND __conan_configurations_to_build ${CMAKE_CONFIGURATION_TYPES})
  endif()

  ternary(system_scientific_libs "True" "False")

  foreach(__conan_build_type ${__conan_configurations_to_build})
    # cmake-format: off
    conan_cmake_run(
      CONANFILE conanfile.py
      SETTINGS  compiler.cppstd=${CMAKE_CXX_STANDARD} 
                build_type=${__conan_build_type}
      INSTALL_FOLDER ${__conan_config_files_dir}
      ENV CC=${CMAKE_C_COMPILER}
      ENV CXX=${CMAKE_CXX_COMPILER}
      BUILD missing
      OPTIONS system_scientific_libs=${__conan_system_scientific_libs}
      UPDATE)
    # cmake-format: on
  endforeach()
endif()

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

if(EXISTS "${__conan_config_files_dir}")
  list(APPEND CMAKE_PREFIX_PATH ${__conan_config_files_dir})
endif()
