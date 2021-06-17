#ifndef APPS_CONJUGATE_GRADIENT_MATRIX_STORAGE_FORMATS
#define APPS_CONJUGATE_GRADIENT_MATRIX_STORAGE_FORMATS

#include "cache_aligned_allocator.h"
#include <vector>
#include <cstdint>

namespace cg {

struct dimensions_t
{
    uint32_t rows{};
    uint32_t cols{};
};

namespace matrix_storage_formats {

    template<typename Ty>
    using cache_aligned_vector = std::vector<Ty, cache_aligned_allocator_t<Ty>>;

    template<typename ValueType, template<typename> typename StorageContainer = cache_aligned_vector>
    struct coo
    {
        dimensions_t dimensions{};
        StorageContainer<ValueType> values;
        StorageContainer<uint32_t> row_indices;
        StorageContainer<uint32_t> col_indices;

        static coo create(dimensions_t dimensions, size_t non_zero_count) {
            return coo{.dimensions = dimensions,
                       .values = decltype(values)(non_zero_count),
                       .row_indices = decltype(row_indices)(non_zero_count),
                       .col_indices = decltype(col_indices)(non_zero_count)};
        }
    };

} // namespace matrix_storage_formats
} // namespace cg
#endif /* APPS_CONJUGATE_GRADIENT_MATRIX_STORAGE_FORMATS */
