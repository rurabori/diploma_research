#ifndef INCLUDE_DIM_MAT_STORAGE_FORMATS_CSR5
#define INCLUDE_DIM_MAT_STORAGE_FORMATS_CSR5

#include <bit>
#include <climits>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <ranges>

#include <dim/bit.h>
#include <dim/mat/storage_formats/base.h>
#include <dim/mat/storage_formats/csr.h>
#include <dim/memory/aligned_allocator.h>
#include <dim/span.h>

#include <omp.h>

#include <emmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>

namespace dim::mat {

template<size_t Sigma, size_t Omega>
struct tile_column_descriptor_t
{
    static constexpr auto tile_size = Sigma * Omega;

    static constexpr auto y_offset_needed = std::bit_width<size_t>(Sigma * Omega);
    static constexpr auto scansum_offset_needed = std::bit_width<size_t>(Omega);
    static constexpr auto bit_flag_needed = Sigma;

    static constexpr auto needed_size = y_offset_needed + scansum_offset_needed + bit_flag_needed;

    static_assert(needed_size <= 64, "we only allow at most 64 bit tile column descriptors");

    using storage_t = std::conditional_t<needed_size <= dim::bit_size<uint32_t>, uint32_t, uint64_t>;

    storage_t y_offset : y_offset_needed = 0;
    storage_t scansum_offset : scansum_offset_needed = 0;
    storage_t bit_flag : bit_flag_needed = 0;

    [[nodiscard]] constexpr auto num_bits_set(bool is_first) const noexcept -> size_t {
        return static_cast<size_t>(std::popcount(bit_flag | static_cast<storage_t>(is_first)));
    }

    [[nodiscard]] auto operator<=>(const tile_column_descriptor_t& other) const noexcept = default;
};

template<size_t Sigma, size_t Omega>
struct tile_descriptor_t
{
    using tile_column_descriptor = tile_column_descriptor_t<Sigma, Omega>;
    tile_column_descriptor columns[Omega];

    // TODO: implement for other combinations.
    [[nodiscard]] auto vectorized() const noexcept requires(Omega == 4 && sizeof(tile_column_descriptor) == 4) {
        constexpr auto y_shift_ammount = 32 - tile_column_descriptor::y_offset_needed;
        constexpr auto scansum_lshift_ammount = y_shift_ammount - tile_column_descriptor::scansum_offset_needed;
        constexpr auto scansum_rshift_amount = 32 - tile_column_descriptor::scansum_offset_needed;
        constexpr auto bit_flag_shift_amount
          = tile_column_descriptor::y_offset_needed + tile_column_descriptor::scansum_offset_needed;

        struct vectorized_impl
        {
            __m128i y_offset;
            __m128i scansum_offset;
            __m128i bit_flag;
        };

        const auto current_desc = _mm_load_si128(reinterpret_cast<const __m128i*>(this));

        return vectorized_impl{
          .y_offset = _mm_srli_epi32(_mm_slli_epi32(current_desc, y_shift_ammount), y_shift_ammount),
          .scansum_offset = _mm_srli_epi32(_mm_slli_epi32(current_desc, scansum_lshift_ammount), scansum_rshift_amount),
          .bit_flag = _mm_srli_epi32(current_desc, bit_flag_shift_amount)};
    }

    [[nodiscard]] auto operator<=>(const tile_descriptor_t& other) const noexcept = default;
};

namespace detail {

    template<typename Ty>
    constexpr bool is_dirty(Ty value) {
        return value & msb<Ty>;
    }

    template<typename Ty>
    constexpr Ty mark_dirty(Ty value) {
        return value | msb<Ty>;
    }

    template<typename Ty>
    constexpr Ty strip_dirty(Ty value) {
        return value & ~msb<Ty>;
    }

    template<bool StripDirty = true, std::ranges::random_access_range Partitions, typename Callable>
    void iterate_partitions(const Partitions& partitions, Callable&& callable) {
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

    template<typename... Ty>
    constexpr int make_permute_seq(Ty... seq) {
        int value{};
        size_t off = 0;
        for (auto pos : std::array{seq...})
            value |= pos << (off++ * 8 / sizeof...(seq));

        return value;
    }

    inline double hsum_avx(__m256d in256d) {
        __m256d hsum = _mm256_add_pd(in256d, _mm256_permute2f128_pd(in256d, in256d, make_permute_seq(1, 0)));

        // NOLINTNEXTLINE - initialization would be dead write.
        double sum;
        _mm_store_sd(&sum, _mm_hadd_pd(_mm256_castpd256_pd128(hsum), _mm256_castpd256_pd128(hsum)));

        return sum;
    }

} // namespace detail

template<std::floating_point ValueType = double, std::signed_integral SignedType = int32_t,
         std::unsigned_integral UnsignedType = uint32_t,
         template<typename> typename StorageContainer = cache_aligned_vector>
struct csr5
{
    // these could be template parameters, but since we're focusing on CPU only,
    // they may be hardcoded.
    static constexpr size_t sigma = 16;
    static constexpr size_t omega = 4;

    using tile_descriptor_type = tile_descriptor_t<sigma, omega>;
    using tile_col_storage = typename tile_descriptor_type::tile_column_descriptor::storage_t;

    // same as CSR
    dimensions_t dimensions;
    StorageContainer<ValueType> vals;
    StorageContainer<UnsignedType> col_idx;
    StorageContainer<UnsignedType> row_ptr;

    size_t tile_count{};
    StorageContainer<UnsignedType> tile_ptr;
    StorageContainer<tile_descriptor_type> tile_desc;
    StorageContainer<UnsignedType> tile_desc_offset_ptr;
    StorageContainer<UnsignedType> tile_desc_offset;

private:
    [[nodiscard]] static auto tile_size() noexcept -> size_t { return sigma * omega; }

    auto set_bit_flag() -> void {
        detail::iterate_partitions(tile_ptr, [&](auto partition_id, auto row_start, auto row_stop) {
            auto&& current_tile_desc = tile_desc[partition_id];

            for (auto rid = row_start; rid <= row_stop; rid++) {
                const auto idx = static_cast<size_t>(row_ptr[rid]);

                // if we aren't in the correct partition, skip.
                if (partition_id != idx / (omega * sigma))
                    continue;

                const auto tile_row = idx % sigma;
                const auto tile_col = (idx / sigma) % omega;

                current_tile_desc.columns[tile_col].bit_flag |= dim::set_rbit<tile_col_storage>(tile_row);
            }
        });
    }

    auto set_tile_descriptor_y_and_segsum_offsets() -> void {
        detail::iterate_partitions(tile_ptr, [&](auto par_id, auto row_start, auto row_stop) {
            // skip empty rows.
            if (row_start == row_stop)
                return;

            auto&& current_descriptor = tile_desc[par_id];

            UnsignedType segn_scan[omega + 1] = {};
            // TODO: check the generated assembly, this will probably not be
            // simd-friendly but the rcount should be faster, check assembly and
            // benchmark.
            tile_col_storage present{1 << omega};

#pragma omp simd
            for (size_t col = 0; col < omega; ++col) {
                const auto& current_column = current_descriptor.columns[col];

                segn_scan[col] = static_cast<UnsignedType>(current_column.num_bits_set(!col));
                present |= segn_scan[col] != 0 ? set_rbit<tile_col_storage>(col) : 0;
            }

            std::exclusive_scan(std::begin(segn_scan), std::end(segn_scan), std::begin(segn_scan), 0);

            const auto has_empty_rows = detail::is_dirty(tile_ptr[par_id]);
            if (has_empty_rows) {
                // total number of empty segments for this partition.
                tile_desc_offset_ptr[par_id] = segn_scan[omega];
                // total number of empty segments for all partitions.
                tile_desc_offset_ptr.back() += segn_scan[omega];
            }

#pragma omp simd
            for (size_t col = 0; col < omega; ++col) {
                auto&& current_column = current_descriptor.columns[col];

                current_column.y_offset = static_cast<tile_col_storage>(segn_scan[col]);
                current_column.scansum_offset = static_cast<tile_col_storage>(
                  has_rbit_set(present, col) ? std::countr_zero(present >> (col + 1)) : 0);
            }
        });
    }

    auto generate_tile_offset() -> void {
        std::exclusive_scan(tile_desc_offset_ptr.begin(), tile_desc_offset_ptr.end(), tile_desc_offset_ptr.begin(), 0);

        tile_desc_offset.resize(tile_desc_offset_ptr[tile_count - 1]);
        detail::iterate_partitions(
          dim::span{tile_ptr}.first(tile_count), [&](const auto par_id, const auto row_start, const auto row_stop) {
              // no empty segments in this tile.
              if (!detail::is_dirty(tile_ptr[par_id]))
                  return;

              const auto offset_ptr = tile_desc_offset_ptr[par_id];
              const auto current_descriptor = tile_desc[par_id];
              const auto rows = dim::span{row_ptr};

#pragma omp unroll
              for (size_t col = 0; col < omega; ++col) {
                  const auto current_column = current_descriptor.columns[col];
                  auto y_offset = current_column.y_offset;

                  const auto col_idx_base = par_id * omega * sigma + col * sigma;
                  for (size_t row = 0; row < sigma; ++row, ++y_offset) {
                      if (!has_rbit_set(current_column.bit_flag | (!col && !row), row))
                          continue;

                      // look for this index in the row table.
                      const auto global_idx = col_idx_base + row;

                      const auto row_data = rows.subspan(row_start, row_stop - row_start);
                      tile_desc_offset[offset_ptr + y_offset]
                        = static_cast<UnsignedType>(detail::upper_bound_idx(row_data, global_idx));
                  }
              }
          });
    }

    auto generate_tile_descriptor() -> void {
        tile_desc.resize(tile_count);
        tile_desc_offset_ptr.resize(tile_ptr.size());

        set_bit_flag();
        set_tile_descriptor_y_and_segsum_offsets();

        // we have some empty segments.
        if (tile_desc_offset_ptr.back())
            generate_tile_offset();
    }

    auto transpose(std::ranges::random_access_range auto& to_transpose) -> void {
        using value_t = std::ranges::range_value_t<decltype(to_transpose)>;

        // last tile is incomplete, thus we'd have access violations if we
        // accessed it.
        const auto tiles = dim::span{tile_ptr}.first(tile_ptr.size() - 1);

        detail::iterate_partitions(tiles, [&](const auto tile_id, const auto row_start, const auto row_stop) {
            if (row_start == row_stop)
                return;

            value_t temp[sigma][omega];

            const auto data_offset = tile_id * omega * sigma;
#pragma omp simd
            for (size_t col = 0; col < omega; ++col) {
                const auto col_offset = col * sigma;
                for (size_t row = 0; row < sigma; ++row)
                    temp[row][col] = to_transpose[data_offset + col_offset + row];
            }

#pragma omp unroll
            for (size_t row = 0; row < sigma; ++row) {
                const auto row_offset = row * omega;
#pragma omp simd
                for (size_t col = 0; col < omega; ++col)
                    to_transpose[data_offset + row_offset + col] = temp[row][col];
            }
        });
    }

    auto transpose() -> void {
        transpose(vals);
        transpose(col_idx);
    }

    auto generate_tile_pointer() {
        tile_ptr.resize(tile_count + 1);

        const auto nnz = vals.size();

#pragma omp parallel for
        for (size_t tile_id = 0; tile_id < tile_ptr.size(); tile_id++) {
            // compute partition boundaries by partition of size sigma * omega,
            // clamp to [0, nnz]
            const auto boundary = static_cast<SignedType>(std::min(tile_id * sigma * omega, nnz));

            tile_ptr[tile_id] = static_cast<UnsignedType>(detail::upper_bound_idx(row_ptr, boundary));
        }

        const auto row_idx = dim::span<const UnsignedType>{row_ptr};
        detail::iterate_partitions(dim::span{tile_ptr}, [&](auto partition_id, auto start, auto stop) {
            if (start == stop)
                return;

            if (detail::is_dirty(row_idx.subspan(start, stop - start)))
                tile_ptr[partition_id] = detail::mark_dirty(start);
        });
    }

    [[nodiscard]] auto tail_partition_start() const noexcept -> UnsignedType {
        return detail::strip_dirty(tile_ptr[tile_count - 1]);
    }

public:
    static auto from_csr(csr<ValueType, StorageContainer> csr) -> csr5 {
        const auto tile_size = sigma * omega;
        const auto num_non_zero = csr.values.size();

        const auto tile_count
          = static_cast<size_t>(std::ceil(static_cast<double>(num_non_zero) / static_cast<double>(tile_size)));

        csr5 val{.dimensions = csr.dimensions,
                 .vals = std::move(csr.values),
                 .col_idx = std::move(csr.col_indices),
                 .row_ptr = std::move(csr.row_start_offsets),
                 .tile_count = tile_count};

        val.generate_tile_pointer();
        val.generate_tile_descriptor();
        val.transpose();

        return val;
    }

    auto load_x(dim::span<const ValueType> x, dim::span<const UnsignedType> column_index_partition,
                size_t offset) const noexcept {
        return _mm256_set_pd(x[column_index_partition[offset + 3]], x[column_index_partition[offset + 2]],
                             x[column_index_partition[offset + 1]], x[column_index_partition[offset]]);
    }

    // TODO: understand this better.
    auto partition_fast_track(dim::span<const ValueType> values, dim::span<const UnsignedType> column_index,
                              dim::span<const ValueType> x, dim::span<ValueType> y, dim::span<ValueType> calibrator,
                              bool direct, UnsignedType start_row_start, UnsignedType row_start, UnsignedType tile_id,
                              UnsignedType stride) const noexcept {
        auto sum256d = _mm256_setzero_pd();

        // fmadd of a single tile where all elements are from a same row.
#pragma unroll
        for (size_t i = 0; i < sigma; i++) {
            const auto base_offset = i * omega;
            sum256d
              = _mm256_fmadd_pd(_mm256_load_pd(&values[base_offset]), load_x(x, column_index, base_offset), sum256d);
        }

        // total sum for this tile.
        const auto sum = detail::hsum_avx(sum256d);

        if (row_start == start_row_start && !direct) {
            // we're in the first tile of chunk assigned to this thread, and the tile is unsealed from top. Need to sync
            // with previous chunk. Throw into the calibrator for now.
            calibrator[tile_id * stride] += sum;
        } else {
            if (direct)
                y[row_start] = sum; // we're the first to access this in our chunk.
            else // TODO: check this case is ok without any synchronization primitives if row spans tiles through
                 // multiple chunks.
                y[row_start] += sum; // not the first to access this in our chunk.
        }
    }

    auto load_tile_info(UnsignedType tile_id) const noexcept {
        static_assert(sizeof(tile_descriptor_type) == 16, "spmv only implemented over 32-bit tile column descriptors");
        const auto current_desc = _mm_load_si128(&tile_desc[tile_id]);
        // const auto y_offset = _mm_slli_epi32(current_desc, )
    }

    auto spmv_full_partitions(dim::span<const ValueType> x, dim::span<ValueType> y,
                              dim::span<ValueType> calibrator) const {
        const auto thread_count = static_cast<size_t>(::omp_get_max_threads());
        const auto chunk
          = static_cast<size_t>(std::ceil(static_cast<double>(tile_count) / static_cast<double>(thread_count)));

        const auto stride = dim::memory::hardware_destructive_interference_size / sizeof(ValueType);
        const auto num_thread_active = static_cast<int>(std::ceil(static_cast<double>(tile_count) / chunk));

#pragma omp parallel
        {
            const auto tid = omp_get_thread_num();
            auto start_row_start = tid < num_thread_active ? strip_dirty(tile_ptr[tid * chunk]) : 0;

            // these need to be aligned to 32 as the intrinsics require it.
            alignas(32) ValueType s_sum[8];
            alignas(32) ValueType s_first_sum[8];
            alignas(32) uint64_t s_cond[8];
            alignas(32) int s_y_idx[16];
            const auto store_to_local_arrays = [&](auto&& y_idx, auto&& sum, auto&& cond) {
                _mm_store_si128(reinterpret_cast<__m128i*>(std::data(s_y_idx)), y_idx);
                _mm256_store_pd(std::data(s_sum), sum);
                _mm256_store_si256(reinterpret_cast<__m256i*>(std::data(s_cond)), cond);
            };

            // TODO: make sense of this +-1 ...
#pragma omp for schedule(static, chunk)
            for (size_t tile_id = 0; tile_id < tile_count - 1; ++tile_id) {
                const auto values = dim::span{vals}.subspan(tile_id * tile_size(), tile_size());
                const auto column_index = dim::span{col_idx}.subspan(tile_id * tile_size(), tile_size());

                auto row_start = tile_ptr[tile_id];
                const auto row_stop = detail::strip_dirty(tile_ptr[tile_id]);

                const auto current_descriptor = tile_desc[tile_id];

                // the tile only has elements from a single row.
                if (row_start == row_stop) {
                    // the row starts at this tile (blue).
                    const auto fast_direct = has_rbit_set(current_descriptor.bit_flag, 0);
                    partition_fast_track(values, column_index, x, y, calibrator, fast_direct, start_row_start,
                                         row_start, tile_id, stride);
                    continue;
                }

                const auto empty_rows = detail::is_dirty(row_start);
                row_start = detail::strip_dirty(row_start);

                auto y_local = y.subspan(row_start + 1);
                const auto store_if_new_row = [&s_cond, &y_local, &s_y_idx, &s_sum](auto idx) {
                    if (!s_cond[idx])
                        return 0;
                    y_local[s_y_idx[idx]] = s_sum[idx];
                    return 1;
                };

                const auto offset_pointer = empty_rows ? tile_desc_offset_ptr[tile_id] : 0;
                const auto compute_y_idx = [this, empty_rows, offset_pointer](auto&& offset) {
                    return empty_rows ? _mm_i32gather_epi32(&tile_desc_offset[offset_pointer], offset, 4) : offset;
                };
            }
        }
    }

    auto spmv(dim::span<const ValueType> x, dim::span<ValueType> y, dim::span<ValueType> calibrator) const {
        spmv_full_partitions(x, y, calibrator);
    }

    auto spmv(dim::span<const ValueType> x, dim::span<ValueType> y) const {
        // TODO: understand the calibrator better.
        StorageContainer<ValueType> calibrator(
          ::omp_get_max_threads() * dim::memory::hardware_destructive_interference_size / sizeof(ValueType),
          ValueType{});

        return spmv(x, y, calibrator);
    }
};

} // namespace dim::mat

#endif /* INCLUDE_DIM_MAT_STORAGE_FORMATS_CSR5 */
