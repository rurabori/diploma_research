#ifndef APPS_CONJUGATE_GRADIENT_TIMED_SECTION
#define APPS_CONJUGATE_GRADIENT_TIMED_SECTION
#include <chrono>
#include <string_view>

#include <fmt/format.h>
#include <fmt/chrono.h>

template<typename Callable>
auto timed_section(Callable&& callable) {
    auto begin = std::chrono::steady_clock::now();
    std::forward<Callable>(callable)();
    return std::chrono::steady_clock::now() - begin;
}

template<typename Callable>
auto report_timed_section(std::string_view name, Callable&& callable) {
    using std::chrono::duration_cast;
    using std::chrono::nanoseconds;

    auto duration = timed_section(std::forward<Callable>(callable));
    fmt::print("{}={}\n", name, duration);
}

#endif /* APPS_CONJUGATE_GRADIENT_TIMED_SECTION */
