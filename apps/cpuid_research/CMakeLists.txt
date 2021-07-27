find_package(cpuinfo REQUIRED)
find_package(fmt REQUIRED)

brr_add_executable(cpuid_research)
target_link_libraries(cpuid_research PRIVATE cpuinfo::cpuinfo fmt::fmt)
