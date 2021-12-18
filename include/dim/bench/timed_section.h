#ifndef INCLUDE_DIM_BENCH_TIMED_SECTION
#define INCLUDE_DIM_BENCH_TIMED_SECTION

#include <dim/bench/stopwatch.h>

#include <concepts>
#include <utility>

namespace dim::bench {

auto section(std::invocable auto&& body) -> second {
    stopwatch sw;
    std::forward<decltype(body)>(body)();
    return sw.elapsed();
}

} // namespace dim::bench

#endif /* INCLUDE_DIM_BENCH_TIMED_SECTION */
