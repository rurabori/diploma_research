#ifndef INCLUDE_DIM_MEMORY_ALIGNED_ALLOCATOR
#define INCLUDE_DIM_MEMORY_ALIGNED_ALLOCATOR

#include <concepts>
#include <cstddef>
#include <new>
#include <vector>

#include <mm_malloc.h>

namespace dim::memory {

#ifdef __cpp_lib_hardware_interference_size
using std::hardware_destructive_interference_size;
#else
constexpr size_t hardware_destructive_interference_size = 64;
#endif

template<typename ValueType>
struct cache_aligned_allocator_t
{
    using value_type = ValueType;

    ValueType* allocate(size_t n) noexcept {
        return reinterpret_cast<ValueType*>(_mm_malloc(n * sizeof(ValueType), hardware_destructive_interference_size));
    }

    void deallocate(ValueType* ptr, size_t /*n*/) noexcept { _mm_free(ptr); }
};

template<typename ValueType1, typename ValueType2>
constexpr bool operator==(const cache_aligned_allocator_t<ValueType1>&,
                          const cache_aligned_allocator_t<ValueType2>&) noexcept {
    return std::same_as<ValueType1, ValueType2>;
}

template<typename ValueType1, typename ValueType2>
constexpr bool operator!=(const cache_aligned_allocator_t<ValueType1>& a,
                          const cache_aligned_allocator_t<ValueType2>& b) noexcept {
    return !(a == b);
}

template<typename Ty>
using cache_aligned_vector = std::vector<Ty, cache_aligned_allocator_t<Ty>>;
} // namespace dim::memory

#endif /* INCLUDE_DIM_MEMORY_ALIGNED_ALLOCATOR */
