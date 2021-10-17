#ifndef INCLUDE_DIM_MAT_STORAGE_FORMATS_BASE
#define INCLUDE_DIM_MAT_STORAGE_FORMATS_BASE

#include <cstdint>
#include <dim/memory/aligned_allocator.h>
#include <vector>

namespace dim::mat {

struct dimensions_t
{
    uint32_t rows{};
    uint32_t cols{};
};

template<typename Ty>
using cache_aligned_vector = std::vector<Ty, dim::memory::cache_aligned_allocator_t<Ty>>;

} // namespace dim::mat

#endif /* INCLUDE_DIM_MAT_STORAGE_FORMATS_BASE */
