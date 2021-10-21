option(enable_coverage "measure code coverage (clang)" OFF)

add_custom_target(coverage)

if(enable_coverage
   AND ("${CMAKE_C_COMPILER_ID}" MATCHES "(Apple)?[Cc]lang"
        OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?[Cc]lang"))
  message(STATUS "Enabling Clang Code Coverage")

  add_compile_options("-fprofile-instr-generate" "-fcoverage-mapping" "-g"
                      "-O0")
  add_link_options("-fprofile-instr-generate" "-fcoverage-mapping")

  function(add_coverage TARGET_NAME)
    if(ARGC GREATER 1)
      list(SUBLIST ARGV 1 -1 ADDITIONAL_ARG)
    endif()

    add_custom_target(
      ccov-${TARGET_NAME}
      COMMAND set LLVM_PROFILE_FILE="$$PWD/${TARGET_NAME}.profraw"
      COMMAND set LLVM_PROFILE_FILE="$$PWD/${TARGET_NAME}.profraw"
      COMMAND cd ${CMAKE_HOME_DIRECTORY} && $<TARGET_FILE:${TARGET_NAME}>
              ${ADDITIONAL_ARG} && cd - || (exit 0)
      COMMAND llvm-profdata-13 merge -sparse ${TARGET_NAME}.profraw -o
              ${TARGET_NAME}.profdata
      # -show-instantiations=true -show-expansions
      COMMAND
        llvm-cov-13 show $<TARGET_FILE:${TARGET_NAME}>
        -instr-profile=${TARGET_NAME}.profdata
        "-ignore-filename-regex=\"(external/.*|tests/.*)\"" -format html
        -output-dir ${CMAKE_BINARY_DIR}/report -show-instantiations=true
        -show-expansions=true -show-line-counts -Xdemangler c++filt -Xdemangler
        -n -show-branches=percent
      COMMAND
        llvm-cov-13 export $<TARGET_FILE:${TARGET_NAME}>
        -instr-profile=${TARGET_NAME}.profdata
        "-ignore-filename-regex=\"(external/.*|tests/.*)\"" -format lcov
        -Xdemangler c++filt -Xdemangler -n > lcov.info
      DEPENDS ${TARGET_NAME}
      BYPRODUCTS ${TARGET_NAME}.profraw ${TARGET_NAME}.profdata
                 ${CMAKE_BINARY_DIR}/report.zip)

    add_dependencies(coverage ccov-${TARGET_NAME})
  endfunction()
else()
  function(add_coverage TARGET_NAME)
    if(ARGC GREATER 1)
      list(SUBLIST ARGV 1 -1 ADDITIONAL_ARG)
    endif()

    add_custom_target(
      ccov-${TARGET_NAME}
      $<TARGET_FILE:${TARGET_NAME}> ${ADDITIONAL_ARG}
      DEPENDS ${TARGET_NAME})
    add_dependencies(coverage ccov-${TARGET_NAME})
  endfunction()
endif()
