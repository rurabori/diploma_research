#ifndef INCLUDE_DIM_MAT_STORAGE_FORMATS_CSR5
#define INCLUDE_DIM_MAT_STORAGE_FORMATS_CSR5

#include <concepts>
#include <cstdint>

#include <dim/mat/storage_formats/base.h>
#include <dim/mat/storage_formats/csr.h>
#include <dim/span.h>

namespace dim::mat {

// TODO: these might be handy for more than just CSR5, move to separate header.
namespace detail {
    /**
     * @brief Computes needed storage for N values (in bits.)
     *
     * @param N the number of values that need to be stored.
     * @return constexpr size_t he amount of bits needed to store N values.F
     */
    template<typename Ty>
    constexpr size_t get_needed_storage(Ty N) {
        size_t retval{1};

        for (Ty base = 2; base < N; base *= 2)
            ++retval;

        return retval;
    }

    template<typename Ty>
    constexpr size_t bit_size = sizeof(Ty) * CHAR_BIT;

    /**
     * @brief Set the bit starting at the most significant bit.
     *
     * @tparam Integral any integral type.
     * @param which the bit to be set (0 means MSB, 1 means MSB - 1 etc.)
     * @return constexpr Integral a value with needed bit set.
     */
    template<typename Integral>
    constexpr Integral set_bit(size_t which) {
        return Integral{1} << bit_size<Integral> - 1 - which;
    }

    /**
     * @brief Checks if a bit is set.
     *
     * @tparam Integral any integral type.
     * @param which the bit to be set (0 means MSB, 1 means MSB - 1 etc.)
     * @return constexpr bool true if bit is set, false otherwise.
     */
    template<typename Integral>
    constexpr bool has_bit_set(Integral value, size_t which) {
        return value & set_bit<Integral>(which);
    }

    template<typename Integral>
    constexpr Integral msb = set_bit<Integral>(0);

    template<std::integral Ty>
    bool is_dirty(Ty value) {
        return value & msb<Ty>;
    }

    template<typename Ty>
    Ty mark_dirty(Ty value) {
        return value | msb<Ty>;
    }

    template<typename Ty>
    Ty strip_dirty(Ty value) {
        return value & ~msb<Ty>;
    }

    template<bool StripDirty = true, typename uiT, typename Callable>
    void iterate_partitions(const dim::span<uiT> partitions, Callable&& callable) {
        constexpr auto conditional_strip = [](auto value) { return StripDirty ? strip_dirty(value) : value; };

#pragma omp parallel for
        for (size_t id = 1; id < partitions.size(); id++) {
            const auto partition_id = id - 1;
            std::forward<Callable>(callable)(partition_id, conditional_strip(partitions[partition_id]),
                                             conditional_strip(partitions[partition_id + 1]));
        }
    }

    template<typename iT>
    bool is_dirty(const dim::span<const iT> row) {
        return std::adjacent_find(row.begin(), row.end()) != row.end();
    }

    template<typename Integral>
    constexpr size_t set_bit_count(const Integral value, const size_t start = 0,
                                   const size_t count = bit_size<Integral>) {
        size_t result{};

#pragma unroll(sizeof(Integral))
        for (size_t i = 0; i < count; ++i)
            result += has_bit_set(value, start + i);

        return result;
    }

    template<typename ValueType>
    constexpr size_t count_consecutive_equal_elements(const dim::span<ValueType> data, const ValueType& example) {
        return static_cast<size_t>(std::distance(
          data.begin(), std::find_if_not(data.begin(), data.end(), [&](auto&& val) { return val == example; })));
    }

    template<typename RangeLike, typename ValueType>
    constexpr size_t upper_bound_idx(const RangeLike& range, const ValueType& val) {
        auto&& it = std::upper_bound(range.begin(), range.end(), val);
        return static_cast<size_t>(std::distance(range.begin(), it) - 1);
    }

} // namespace detail

template<std::floating_point ValueType = double, std::signed_integral SignedType = int32_t,
         std::unsigned_integral UnsignedType = uint32_t,
         template<typename> typename StorageContainer = cache_aligned_vector>
struct csr5
{
    // same as CSR
    dimensions_t dimensions;
    StorageContainer<ValueType> vals;
    StorageContainer<UnsignedType> col_idx;
    StorageContainer<UnsignedType> row_ptr;

    // TODO: template parameters?
    size_t sigma;
    size_t omega;

    size_t tile_count;
    // TODO: does the descriptor need to be uint32_t? Might benefit from being 64-bit.
    StorageContainer<UnsignedType> tile_ptr;
    StorageContainer<UnsignedType> tile_desc;

    auto generate_tile_pointer() {
        tile_ptr.resize(tile_count + 1);

        const auto nnz = vals.size();

#pragma omp parallel for
        for (size_t global_id = 0; global_id < tile_ptr.size(); global_id++) {
            // compute partition boundaries by partition of size sigma * omega,
            // clamp to [0, nnz]
            const auto boundary = static_cast<SignedType>(std::min(global_id * sigma * omega, nnz));

            tile_ptr[global_id] = static_cast<UnsignedType>(detail::upper_bound_idx(row_ptr, boundary));
        }

        const auto row_idx = dim::span<const UnsignedType>{row_ptr};
        detail::iterate_partitions(dim::span{tile_ptr}, [&](auto partition_id, auto start, auto stop) {
            if (start == stop)
                return;

            if (detail::is_dirty(row_idx.subspan(start, stop - start)))
                tile_ptr[partition_id] = detail::mark_dirty(start);
        });
    }

    static auto from_csr(csr<ValueType, StorageContainer> csr, size_t sigma = 16, size_t omega = 4) -> csr5 {
        const auto tile_size = sigma * omega;
        const auto num_non_zero = csr.values.size();

        const auto tile_count
          = static_cast<size_t>(std::ceil(static_cast<double>(num_non_zero) / static_cast<double>(tile_size)));

        csr5 val{.dimensions = csr.dimensions,
                 .vals = std::move(csr.values),
                 .col_idx = std::move(csr.col_indices),
                 .row_ptr = std::move(csr.row_start_offsets),
                 .sigma = sigma,
                 .omega = omega,
                 .tile_count = tile_count};

        val.generate_tile_pointer();

        return val;
    }
};

} // namespace dim::mat

#endif /* INCLUDE_DIM_MAT_STORAGE_FORMATS_CSR5 */
