#ifndef INCLUDE_DIM_BENCH_STOPWATCH
#define INCLUDE_DIM_BENCH_STOPWATCH

#include <chrono>

namespace dim::bench {

// seconds but backed by a double.
using second = std::chrono::duration<double>;

class stopwatch
{
    using clock_t = std::chrono::steady_clock;

    clock_t::time_point _start{clock_t::now()};

public:
    auto reset() noexcept -> void { _start = clock_t::now(); }
    [[nodiscard]] auto elapsed() const noexcept -> second {
        return std::chrono::duration_cast<second>(clock_t::now() - _start);
    }
};

} // namespace dim::bench

#endif /* INCLUDE_DIM_BENCH_STOPWATCH */
