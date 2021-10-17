#ifndef INCLUDE_DIM_MAT_STORAGE_FORMATS_COO
#define INCLUDE_DIM_MAT_STORAGE_FORMATS_COO

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include <dim/mat/storage_formats/base.h>
#include <dim/memory/aligned_allocator.h>

namespace dim::mat {

template<typename ValueType, template<typename> typename StorageContainer = cache_aligned_vector>
struct coo
{
    dimensions_t dimensions{};
    StorageContainer<ValueType> values;
    StorageContainer<uint32_t> row_indices;
    StorageContainer<uint32_t> col_indices;
    bool symmetric{false};

    coo(dimensions_t dimensions, size_t non_zero_count, bool symmetric = false)
      : dimensions{dimensions},
        values(non_zero_count),
        row_indices(non_zero_count),
        col_indices(non_zero_count),
        symmetric{symmetric} {}

    template<typename /*std::invocable<ValueType, uint32_t, uint32_t>*/ Callable>
    void iterate(Callable&& callable) const noexcept {
        for (size_t i = 0; i < values.size(); ++i) {
            const auto row = row_indices[i];
            const auto col = col_indices[i];

            // send the values in.
            callable(values[i], row, col);

            // for symmetric matrices, we also need to invoke with the reverse.
            if (symmetric && row != col)
                callable(values[i], col, row);
        }
    }
};

} // namespace dim::mat

#endif /* INCLUDE_DIM_MAT_STORAGE_FORMATS_COO */
