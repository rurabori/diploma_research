#ifndef INCLUDE_DIM_SIMPLE_MAIN
#define INCLUDE_DIM_SIMPLE_MAIN

#if !__has_include("version.h")
#error "This file may only be used with applications using brr versioning header."
#endif

#include "version.h"

#include <fmt/format.h>

#include <structopt/app.hpp>
#include <structopt/exception.hpp>

#include <spdlog/spdlog.h>

// NOLINTNEXTLINE - these war crimes are the simplest way to make a simple application.
#define DIM_MAIN(ArgumentsType)                                                    \
    int main(int argc, char* argv[]) try {                                         \
        auto app = structopt::app(brr::app_info.full_name, brr::app_info.version); \
                                                                                   \
        return main_impl(app.parse<ArgumentsType>(argc, argv));                    \
    } catch (const structopt::exception& e) {                                      \
        spdlog::critical("{}", e.what());                                          \
        fmt::print(stderr, "{}", e.help());                                        \
        return 1;                                                                  \
    } catch (const std::exception& e) {                                            \
        spdlog::critical("{}", e.what());                                          \
        return 2;                                                                  \
    }                                                                              \
    int main(int, char*[])

#endif /* INCLUDE_DIM_SIMPLE_MAIN */
