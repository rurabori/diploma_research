#include <span>

#include <cpuinfo.h>
#include <fmt/core.h>

int main(int argc, const char* argv[]) {
    cpuinfo_initialize();

    const auto packages = std::span{cpuinfo_get_packages(), cpuinfo_get_packages_count()};
    for (auto&& package : packages) {
        fmt::print("Package {}, avx2 supported: {}\n", package.name, cpuinfo_has_x86_avx2());
    }

    cpuinfo_deinitialize();
    return 0;
}