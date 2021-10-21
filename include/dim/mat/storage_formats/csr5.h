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
#include <dim/simd.h>
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
    using col_storage = typename tile_column_descriptor::storage_t;

    struct scansum_and_present_t
    {
        col_storage segn_scan[Omega + 1];
        col_storage present{1 << Omega};
    };

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

            [[nodiscard]] __m256i get_local_bit(int which) const noexcept {
                return _mm256_and_si256(_mm256_cvtepu32_epi64(_mm_srli_epi32(bit_flag, which)),
                                        _mm256_set1_epi64x(0x1));
            }
        };

        const auto current_desc = _mm_load_si128(reinterpret_cast<const __m128i*>(this));

        return vectorized_impl{
          .y_offset = _mm_srli_epi32(_mm_slli_epi32(current_desc, y_shift_ammount), y_shift_ammount),
          .scansum_offset = _mm_srli_epi32(_mm_slli_epi32(current_desc, scansum_lshift_ammount), scansum_rshift_amount),
          .bit_flag = _mm_srli_epi32(current_desc, bit_flag_shift_amount)};
    }

    auto iterate_columns(std::invocable<size_t, tile_column_descriptor> auto&& body) const {
#pragma omp simd
        for (size_t col = 0; col < Omega; ++col) {
            body(col, columns[col]);
        }
    }

    auto iterate_columns(std::invocable<size_t, tile_column_descriptor&> auto&& body) {
#pragma omp simd
        for (size_t col = 0; col < Omega; ++col) {
            body(col, columns[col]);
        }
    }

    [[nodiscard]] auto generate_scansum_and_present() const noexcept -> scansum_and_present_t {
        scansum_and_present_t retval;

        iterate_columns([&](size_t col_idx, auto col) {
            retval.segn_scan[col_idx] = static_cast<col_storage>(col.num_bits_set(!col_idx));
            retval.present |= retval.segn_scan[col_idx] != 0 ? set_rbit<col_storage>(col_idx) : 0;
        });

        std::exclusive_scan(std::begin(retval.segn_scan), std::end(retval.segn_scan), std::begin(retval.segn_scan), 0);

        return retval;
    }

    auto set_scansum_and_y_offsets() noexcept -> size_t {
        const auto desc = generate_scansum_and_present();

        iterate_columns([&](size_t col_idx, auto& col) {
            col.y_offset = static_cast<col_storage>(desc.segn_scan[col_idx]);
            col.scansum_offset = static_cast<col_storage>(
              has_rbit_set(desc.present, col_idx) ? std::countr_zero(desc.present >> (col_idx + 1)) : 0);
        });

        return desc.segn_scan[Omega];
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
    void iterate_tiles(const Partitions& partitions, Callable&& callable) {
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

    inline auto fast_segmented_sum(__m256d vec, __m128i segsum_offset) noexcept -> __m256d {
        const auto zero_last_mask = _mm256_castsi256_pd(
          _mm256_set_epi64x(0, all_bits_set<int64_t>, all_bits_set<int64_t>, all_bits_set<int64_t>));

        // rotate one to the left, to match
        vec = _mm256_permute4x64_pd(vec, simd::make_permute_seq(1, 2, 3, 0));
        // zero out the first element of the summed vector.
        vec = _mm256_and_pd(zero_last_mask, vec);

        const auto tmp_sum256d = vec;
        // inclusive prefix scan.
        vec = simd::hscan_avx(vec);

        // vec[i] = vec[i + seg_offset[i]] - vec[i] + tmp[i]
        return _mm256_add_pd(_mm256_sub_pd(simd::shuffle_relative(vec, segsum_offset), vec), tmp_sum256d);
    }

    inline auto compute_last_sum(__m256d sum, __m256d first_sum, __m128i segsum_offset, __m256i start,
                                 __m256i stop) noexcept -> __m256d {
        // for lane i:
        // sum[i] = col_uncapped_from_top[i] ? first_sum[i] : 0.
        const auto col_uncapped_from_top = _mm256_cmpeq_epi64(start, _mm256_set1_epi64x(0x1));
        // leaving only sums of columns which start by red section untouched.
        first_sum = _mm256_and_pd(_mm256_castsi256_pd(col_uncapped_from_top), first_sum);

        // sums of red parts for next scansum_offset columns.
        const auto next_prefix_sum = detail::fast_segmented_sum(first_sum, segsum_offset);

        // no row started and no row ended in the column (start256i is 1 if first bit flag in column isn't
        // set, and stop is 0 if no other bit was set).
        const auto not_red_column = _mm256_cmpgt_epi64(start, stop);

        // only add parts which had any rows starting here (bitflag set).
        return _mm256_add_pd(sum, _mm256_andnot_pd(_mm256_castsi256_pd(not_red_column), next_prefix_sum));
    };

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
        detail::iterate_tiles(tile_ptr, [&](auto tile_id, auto row_start, auto row_stop) {
            auto&& current_tile_desc = tile_desc[tile_id];

            // TODO: this is a bit wasteful if we have long rows.
            for (auto rid = row_start; rid <= row_stop; rid++) {
                const auto idx = static_cast<size_t>(row_ptr[rid]);

                // if we aren't in the correct tile yet, skip.
                if (tile_id != idx / (omega * sigma))
                    continue;

                const auto tile_row = idx % sigma;
                const auto tile_col = (idx / sigma) % omega;

                current_tile_desc.columns[tile_col].bit_flag |= dim::set_rbit<tile_col_storage>(tile_row);
            }
        });
    }

    auto set_tile_descriptor_y_and_segsum_offsets() -> void {
        detail::iterate_tiles(tile_ptr, [&](auto tile_id, auto row_start, auto row_stop) {
            // skip tiles which have all elements in the same row (they are fast tracked).
            if (row_start == row_stop)
                return;

            const auto num_set = static_cast<UnsignedType>(tile_desc[tile_id].set_scansum_and_y_offsets());

            const auto has_empty_rows = detail::is_dirty(tile_ptr[tile_id]);
            if (has_empty_rows) {
                // total number of empty segments for this partition.
                tile_desc_offset_ptr[tile_id] = num_set;
                // total number of empty segments for all partitions.
                tile_desc_offset_ptr.back() += num_set;
            }
        });
    }

    auto generate_tile_offset() -> void {
        std::exclusive_scan(tile_desc_offset_ptr.begin(), tile_desc_offset_ptr.end(), tile_desc_offset_ptr.begin(), 0);

        tile_desc_offset.resize(tile_desc_offset_ptr.back());
        detail::iterate_tiles(tile_ptr, [&](const auto tile_id, const auto row_start, const auto row_stop) {
            // no empty segments in this tile.
            if (!detail::is_dirty(tile_ptr[tile_id]))
                return;

            const auto offset_ptr = tile_desc_offset_ptr[tile_id];
            auto current_descriptor = tile_desc[tile_id];
            current_descriptor.columns[0].bit_flag |= 1;

            // this is inclusive.
            const auto rows = dim::span{row_ptr}.subspan(row_start, row_stop - row_start + 1);
            current_descriptor.iterate_columns([&](size_t col, auto desc) {
                // skip columns which don't have any rows starting in them altogether.
                if (!desc.bit_flag)
                    return;

                auto y_offset = desc.y_offset;
                const auto col_idx_base = tile_id * omega * sigma + col * sigma;

                for (size_t row = 0; row < sigma; ++row) {
                    if (!has_rbit_set(desc.bit_flag, row))
                        continue;

                    // look for this index in the row table.
                    const auto global_idx = col_idx_base + row;

                    tile_desc_offset[offset_ptr + y_offset]
                      = static_cast<UnsignedType>(detail::upper_bound_idx(rows, global_idx));

                    ++y_offset;
                }
            });
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

        detail::iterate_tiles(tiles, [&](const auto tile_id, const auto row_start, const auto row_stop) {
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
        detail::iterate_tiles(dim::span{tile_ptr}, [&](auto partition_id, auto start, auto stop) {
            if (start == stop)
                return;

            if (detail::is_dirty(row_idx.subspan(start, stop - start)))
                tile_ptr[partition_id] = detail::mark_dirty(start);
        });
    }

public:
    static auto from_csr(csr<ValueType, StorageContainer> csr) -> csr5 {
        const auto tile_size = sigma * omega;
        const auto num_non_zero = csr.values.size();

        const auto tile_count
          = static_cast<size_t>(std::floor(static_cast<double>(num_non_zero) / static_cast<double>(tile_size)));

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

    [[nodiscard]] auto tail_partition_start() const noexcept -> UnsignedType {
        return detail::strip_dirty(tile_ptr.back());
    }

    auto load_x(dim::span<const ValueType> x, dim::span<const UnsignedType> column_index_partition,
                size_t offset) const noexcept {
        return _mm256_set_pd(x[column_index_partition[offset + 3]], x[column_index_partition[offset + 2]],
                             x[column_index_partition[offset + 1]], x[column_index_partition[offset]]);
    }

    struct spmv_data_t
    {
        dim::span<const ValueType> x;
        dim::span<ValueType> y;
        dim::span<ValueType> calibrator;
    };

    struct spmv_thread_data
    {
        static constexpr auto stride = dim::memory::hardware_destructive_interference_size / sizeof(ValueType);

        // these need to be aligned to 32 as the intrinsics require it.
        const spmv_data_t& data;
        int tid;
        UnsignedType start_row_start;

        // these need to be aligned to 32 as the intrinsics require it.
        alignas(32) ValueType sum[8];
        alignas(32) ValueType first_sum[8];
        alignas(32) uint64_t cond[8];
        alignas(32) int y_idx[16];

        auto store(__m128i y_index, __m256d lsum, __m256i lcond) {
            _mm_store_si128(reinterpret_cast<__m128i*>(std::data(y_idx)), y_index);
            _mm256_store_pd(std::data(sum), lsum);
            _mm256_store_si256(reinterpret_cast<__m256i*>(std::data(cond)), lcond);
        }

        auto maybe_store(dim::span<ValueType> y, size_t idx) const noexcept {
            if (!cond[idx])
                return 0;

            y[y_idx[idx]] = sum[idx];
            return 1;
        }

        [[nodiscard]] auto write_to_y(dim::span<ValueType> y) const noexcept -> __m128i {
            return _mm_set_epi32(maybe_store(y, 3), maybe_store(y, 2), maybe_store(y, 1), maybe_store(y, 0));
        }

        auto store_to_row_start(UnsignedType row_start, ValueType val, bool direct) const -> void {
            if (row_start == start_row_start && !direct) {
                // we're in the first tile of chunk assigned to this thread, and the tile is unsealed from top. Need to
                // sync with previous chunk. Throw into the calibrator for now.
                data.calibrator[tid * stride] += val;
                return;
            }

            if (direct)
                data.y[row_start] = val; // we're the first to access this in our chunk.
            else // TODO: check this case is ok without any synchronization primitives if row spans tiles through
                 // multiple chunks.
                data.y[row_start] += val; // not the first to access this in our chunk.
        }
    };

    // TODO: understand this better.
    auto partition_fast_track(dim::span<const ValueType> values, dim::span<const UnsignedType> column_index,
                              dim::span<const ValueType> x) const noexcept -> ValueType {
        auto sum256d = _mm256_setzero_pd();

        // fmadd of a single tile where all elements are from a same row.
#pragma unroll
        for (size_t i = 0; i < sigma; i++) {
            const auto base_offset = i * omega;
            sum256d
              = _mm256_fmadd_pd(_mm256_load_pd(&values[base_offset]), load_x(x, column_index, base_offset), sum256d);
        }

        // total sum for this tile.
        return simd::hsum_avx(sum256d);
    }

    auto spmv_full_partitions(spmv_data_t spmv_data) const {
        const auto thread_count = static_cast<size_t>(::omp_get_max_threads());
        const auto chunk
          = static_cast<size_t>(std::ceil(static_cast<double>(tile_count) / static_cast<double>(thread_count)));

        const auto num_thread_active = static_cast<int>(std::ceil(static_cast<double>(tile_count) / chunk));

#pragma omp parallel
        {
            const auto tid = omp_get_thread_num();

            spmv_thread_data thread_data{.data = spmv_data,
                                         .tid = tid,
                                         .start_row_start
                                         = tid < num_thread_active ? detail::strip_dirty(tile_ptr[tid * chunk]) : 0};

            // TODO: make sense of this +-1 ...
#pragma omp for schedule(static, chunk)
            for (size_t tile_id = 0; tile_id < tile_count - 1; ++tile_id) {
                const auto values = dim::span{vals}.subspan(tile_id * tile_size(), tile_size());
                const auto column_index = dim::span{col_idx}.subspan(tile_id * tile_size(), tile_size());

                auto row_start = tile_ptr[tile_id];
                const auto row_stop = detail::strip_dirty(tile_ptr[tile_id]);

                const auto current_descriptor = tile_desc[tile_id];

                const auto fast_direct = has_rbit_set(current_descriptor.columns[0].bit_flag, 0);

                // the tile only has elements from a single row.
                if (row_start == row_stop) {
                    // the row starts at this tile (blue).
                    thread_data.store_to_row_start(row_start, partition_fast_track(values, column_index, spmv_data.x),
                                                   fast_direct);
                    continue;
                }

                const auto empty_rows = detail::is_dirty(row_start);
                row_start = detail::strip_dirty(row_start);

                // TODO: careful with the offsets here. We never write directly to the first element.
                auto y_local = spmv_data.y.subspan(row_start);

                const auto offset_pointer = empty_rows ? tile_desc_offset_ptr[tile_id] : 0;
                const auto compute_y_idx = [this, empty_rows, offset_pointer](auto&& offset) {
                    return empty_rows ? _mm_i32gather_epi32(&tile_desc_offset[offset_pointer], offset, 4) : offset;
                };

                auto vec = current_descriptor.vectorized();

                // remember if the first element of this partition is the first element of a new row
                auto local_bit256i = vec.get_local_bit(0);
                _mm256_store_si256(reinterpret_cast<__m256i*>(std::data(thread_data.cond)), local_bit256i);

                // remember if the first element of the first partition of the current thread is the first element of a
                // new row
                const bool first_all_direct = tile_id == tid * chunk && fast_direct;

                // set the 0th bit of the descriptor to 1.
                // NOLINTNEXTLINE - clang doesn't support bit_cast yet.
                vec.bit_flag = _mm_or_si128(vec.bit_flag, _mm_set_epi32(0, 0, 0, 1));

                // load bits for next 4 reductions.
                local_bit256i = vec.get_local_bit(0);

                // start256i = !local_bit256i (speaking in bools). Meaning it's true if no row started here.
                auto start256i = _mm256_sub_epi64(_mm256_set1_epi64x(0x1), local_bit256i);

                auto stop256i = _mm256_setzero_si256();
                auto any_row_started = _mm256_and_si256(local_bit256i, _mm256_set_epi64x(0x1, 0x1, 0x1, 0));

                // do the first sum.
                auto value256d = _mm256_load_pd(values.data());
                auto x256d = load_x(spmv_data.x, column_index, 0);
                auto sum256d = _mm256_mul_pd(value256d, x256d);

                // move the first column y_offset by one, that write is handled at the end of this function.
                vec.y_offset = _mm_add_epi32(vec.y_offset, _mm_set_epi32(0, 0, 0, 1));

                // step 1. thread-level seg sum
                auto first_sum256d = _mm256_setzero_pd();
                for (UnsignedType i = 1; i < sigma; ++i) {
                    local_bit256i = vec.get_local_bit(i);

                    // any of the bits mark start of a new row.
                    if (simd::any_bit_set(local_bit256i)) {
                        // if empty rows we need to use empty_offset[y_offset]
                        auto y_idx128i = compute_y_idx(vec.y_offset);

                        // green section capped from both sides.
                        const auto full_row = _mm256_and_si256(any_row_started, local_bit256i);
                        thread_data.store(y_idx128i, sum256d, full_row);

                        // if any of the lanes stored, we need to increase its index (new row == new output in Y).
                        vec.y_offset = _mm_add_epi32(vec.y_offset, thread_data.write_to_y(y_local));

                        const auto localbit_mask = _mm256_cmpeq_epi64(local_bit256i, _mm256_set1_epi64x(0x1));
                        const auto row_active_mask = _mm256_cmpeq_epi64(any_row_started, _mm256_set1_epi64x(0x1));

                        // for lane i:
                        // first_sum[i] = !row_active[i] && localbit[i] ? sum256d : first_sum256d.
                        // thus, if we're at the end of a red section, we store the sum here to reuse after the loop is
                        // done (no other red section can occur in this lane).
                        const auto mask = _mm256_andnot_si256(row_active_mask, localbit_mask);
                        first_sum256d = simd::merge_vec(first_sum256d, sum256d, mask);

                        // zero out the parts of sum which have localbit set (AKA starting new row).
                        sum256d = _mm256_andnot_pd(_mm256_castsi256_pd(localbit_mask), sum256d);
                        // set direct to know if we've started a row yet.
                        any_row_started = _mm256_or_si256(any_row_started, local_bit256i);
                        // increase count of total rows started.
                        stop256i = _mm256_add_epi64(stop256i, local_bit256i);
                    }

                    // process the next row in bitmap.
                    x256d = load_x(spmv_data.x, column_index, i * omega);
                    value256d = _mm256_load_pd(&values[i * omega]);
                    sum256d = _mm256_fmadd_pd(value256d, x256d, sum256d);
                }

                // for lane i:
                // first_sum[i] = row_active[i] ? first_sum256[i] : sum256[i]
                // thus if any row has started in lane i, first_sum is unchanged, but if none started,
                // we have a full red column and first_sum256d is the actual sum so far.
                const auto any_row_active = _mm256_cmpeq_epi64(any_row_started, _mm256_set1_epi64x(0x1));
                // contains the potential sum of elements in red part of each lane.
                first_sum256d = simd::merge_vec(sum256d, first_sum256d, any_row_active);

                const auto last_sum
                  = detail::compute_last_sum(sum256d, first_sum256d, vec.scansum_offset, start256i, stop256i);

                auto y_idx128i = compute_y_idx(vec.y_offset);
                thread_data.store(y_idx128i, last_sum, any_row_started);
                (void)thread_data.write_to_y(y_local);

                if (thread_data.cond[0])
                    _mm256_store_pd(thread_data.first_sum, first_sum256d);

                thread_data.store_to_row_start(
                  row_start, thread_data.cond[0] ? thread_data.first_sum[0] : thread_data.sum[0], first_all_direct);
            }
        }
    }

    auto spmv(dim::span<const ValueType> x, dim::span<ValueType> y, dim::span<ValueType> calibrator) const {
        spmv_full_partitions(spmv_data_t{.x = x, .y = y, .calibrator = calibrator});
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
