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
#include <dim/opt.h>
#include <dim/simd.h>
#include <dim/span.h>

#include <omp.h>

#include <emmintrin.h>
#include <immintrin.h>
#include <stdexcept>
#include <xmmintrin.h>

namespace dim::mat {

template<typename DefType>
struct tile_column_descriptor_t
{
    using def_t = DefType;
    using storage_t = typename def_t::storage_t;

    storage_t y_offset : def_t::y_offset = 0;
    storage_t scansum_offset : def_t::scansum_offset = 0;
    storage_t bit_flag : def_t::bit_flag = 0;

    [[nodiscard]] constexpr auto num_bits_set(bool is_first = false) const noexcept -> size_t {
        return static_cast<size_t>(std::popcount(bit_flag | static_cast<storage_t>(is_first)));
    }

    [[nodiscard]] auto operator<=>(const tile_column_descriptor_t& other) const noexcept = default;
};

template<size_t Sigma, size_t Omega>
struct tile_descriptor_t
{
    struct storage_def_t
    {
        static constexpr auto y_offset = std::bit_width<size_t>(Sigma * Omega);
        static constexpr auto scansum_offset = std::bit_width<size_t>(Omega);
        static constexpr auto bit_flag = Sigma;
        static constexpr auto needed_size = y_offset + scansum_offset + bit_flag;

        static_assert(needed_size <= 64, "we only allow at most 64 bit tile column descriptors");

        using storage_t = std::conditional_t<needed_size <= dim::bit_size<uint32_t>, uint32_t, uint64_t>;
    };

    using tile_column_descriptor = tile_column_descriptor_t<storage_def_t>;
    using col_storage = typename tile_column_descriptor::storage_t;

    struct scansum_and_present_t
    {
        col_storage segn_scan[Omega + 1];
        col_storage present{1 << Omega};
    };

    static constexpr auto num_cols = Omega;
    static constexpr auto num_rows = Sigma;
    static constexpr auto block_size = Sigma * Omega;

    tile_column_descriptor columns[Omega];

    [[nodiscard]] auto vectorized() const noexcept requires(Omega == 4 && sizeof(tile_column_descriptor) == 4) {
        constexpr auto y_shift_amount = 32 - storage_def_t::y_offset;
        constexpr auto scansum_lshift_amount = y_shift_amount - storage_def_t::scansum_offset;
        constexpr auto scansum_rshift_amount = 32 - storage_def_t::scansum_offset;
        constexpr auto bit_flag_lshift_amount = scansum_lshift_amount - storage_def_t::bit_flag;
        constexpr auto bit_flag_rshift_amount = 32 - storage_def_t::bit_flag;

        struct vectorized_impl
        {
            __m128i y_offset;
            __m128i scansum_offset;
            __m128i bit_flag;

            [[nodiscard]] __m256i get_local_bit(size_t which) const noexcept {
                return _mm256_and_si256(_mm256_cvtepu32_epi64(_mm_srli_epi32(bit_flag, static_cast<int>(which))),
                                        _mm256_set1_epi64x(0x1));
            }
        };

        const auto* tmp = reinterpret_cast<const int*>(&columns);
        const auto current_desc = _mm_set_epi32(tmp[3], tmp[2], tmp[1], tmp[0]);

        return vectorized_impl{
          .y_offset = _mm_srli_epi32(_mm_slli_epi32(current_desc, y_shift_amount), y_shift_amount),
          .scansum_offset = _mm_srli_epi32(_mm_slli_epi32(current_desc, scansum_lshift_amount), scansum_rshift_amount),
          .bit_flag = _mm_srli_epi32(_mm_slli_epi32(current_desc, bit_flag_lshift_amount), bit_flag_rshift_amount)};
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

    [[nodiscard]] auto operator==(const tile_descriptor_t& other) const noexcept {
        return std::equal(std::begin(columns), std::end(columns), std::begin(other.columns));
    }

    [[nodiscard]] auto is_uncapped() const noexcept -> bool { return !has_rbit_set(columns[0].bit_flag, 0); }
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

    template<bool StripDirty = true, std::ranges::random_access_range Tiles, typename Callable>
    void iterate_tiles(const Tiles& tiles, Callable&& callable) {
        constexpr auto conditional_strip = [](auto value) { return StripDirty ? strip_dirty(value) : value; };

#pragma omp parallel for
        for (size_t id = 1; id < tiles.size(); id++) {
            const auto tile_id = id - 1;
            std::forward<Callable>(callable)(tile_id, conditional_strip(tiles[tile_id].raw),
                                             conditional_strip(tiles[tile_id + 1].raw));
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

        const auto b = _mm256_andnot_pd(_mm256_castsi256_pd(not_red_column), next_prefix_sum);
        // only add parts which had any rows starting here (bitflag set).
        return _mm256_add_pd(sum, b);
    }

} // namespace detail

template<typename StorageType>
struct tile_ptr_t
{
    using storage_t = StorageType;

    StorageType raw;

    auto mark_dirty() noexcept -> void { raw = detail::mark_dirty(raw); }
    auto relative(StorageType to) const noexcept -> tile_ptr_t { return {raw - to}; }
    [[nodiscard]] auto is_dirty() const noexcept -> bool { return detail::is_dirty(raw); }
    [[nodiscard]] auto idx() const noexcept -> StorageType { return detail::strip_dirty(raw); }

    friend auto operator<=>(const tile_ptr_t&, const tile_ptr_t&) noexcept = default;
};

template<std::floating_point ValueType = double, size_t Sigma = 16, size_t Omega = 4,
         std::signed_integral SignedType = int32_t, std::unsigned_integral UnsignedType = uint32_t,
         template<typename> typename StorageContainer = cache_aligned_vector>
struct csr5
{
    // these could be template parameters, but since we're focusing on CPU only,
    // they may be hardcoded.
    static constexpr size_t sigma = Sigma;
    static constexpr size_t omega = Omega;

    [[nodiscard]] constexpr static auto tile_size() noexcept -> size_t { return sigma * omega; }

    using tile_descriptor_type = tile_descriptor_t<sigma, omega>;
    using tile_col_storage = typename tile_descriptor_type::tile_column_descriptor::storage_t;
    using tile_ptr_type = tile_ptr_t<UnsignedType>;

    struct csr5_info_t
    {
        size_t tile_count{};
        StorageContainer<tile_ptr_type> tile_ptr{};
        StorageContainer<tile_descriptor_type> tile_desc{};
        StorageContainer<UnsignedType> tile_desc_offset_ptr{};
        StorageContainer<UnsignedType> tile_desc_offset{};

        auto iterate_tiles(std::invocable<size_t, size_t, size_t> auto&& callable) const {
            detail::iterate_tiles(tile_ptr, callable);
        }

        [[nodiscard]] auto empty_offset_for(size_t tile_id) const noexcept -> UnsignedType {
            return tile_desc_offset_ptr[tile_id] - tile_desc_offset_ptr.front();
        }

        [[nodiscard]] auto first_row_idx() const noexcept -> UnsignedType { return tile_ptr.front().idx(); }
        [[nodiscard]] auto last_row_idx() const noexcept -> UnsignedType {
            assert(tile_count);

            const auto tile_id = tile_count - 1;
            const auto ptr = tile_ptr[tile_id];

            const auto last_col = tile_desc[tile_id].columns[Omega - 1];

            // y_idx is always > 0 (bit_flag[0, 0] is always 1 implicitly) + we add number of sections started - 1
            // because y_index is telling us index of next section that would start after it.
            const auto relative_offset = last_col.y_offset + last_col.num_bits_set() - 1;

            if (!ptr.is_dirty())
                return ptr.idx() + static_cast<UnsignedType>(relative_offset);

            return ptr.idx() + tile_desc_offset[empty_offset_for(tile_id) + relative_offset];
        }

    private:
        auto set_bit_flag(std::span<const UnsignedType> row_ptr_) -> void {
            iterate_tiles([&](auto tile_id, auto row_start, auto row_stop) {
                auto&& current_tile_desc = tile_desc[tile_id];

                for (auto rid = row_start; rid <= row_stop; rid++) {
                    const auto idx = static_cast<size_t>(row_ptr_[rid]);

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
            iterate_tiles([&](auto tile_id, auto row_start, auto row_stop) {
                // skip tiles which have all elements in the same row (they are fast tracked).
                if (row_start == row_stop)
                    return;

                const auto num_set = static_cast<UnsignedType>(tile_desc[tile_id].set_scansum_and_y_offsets());

                const auto has_empty_rows = tile_ptr[tile_id].is_dirty();
                if (has_empty_rows) {
                    // y_idx for some columns of the tile may be 1 bigger than number of segments it contains, may be
                    // avoided by rearchitecturing the code a bit.
                    const auto needed_offsets = num_set + 1;
                    // total number of empty segments for this partition.
                    tile_desc_offset_ptr[tile_id] = needed_offsets;
                    // total number of empty segments for all partitions.
                    tile_desc_offset_ptr.back() += needed_offsets;
                }
            });
        }

        auto generate_tile_offset(std::span<const UnsignedType> row_ptr_) -> void {
            std::exclusive_scan(tile_desc_offset_ptr.begin(), tile_desc_offset_ptr.end(), tile_desc_offset_ptr.begin(),
                                0);
            tile_desc_offset.resize(tile_desc_offset_ptr.back());

            iterate_tiles([&](const auto tile_id, const auto row_start, const auto row_stop) {
                // no empty segments in this tile.
                if (!tile_ptr[tile_id].is_dirty())
                    return;

                const auto offset_ptr = tile_desc_offset_ptr[tile_id];
                auto current_descriptor = tile_desc[tile_id];
                current_descriptor.columns[0].bit_flag |= 1;

                // this is inclusive.
                const auto rows = row_ptr_.subspan(row_start, row_stop - row_start + 1);
                current_descriptor.iterate_columns([&](size_t col, auto desc) {
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

        auto generate_tile_descriptor(std::span<const UnsignedType> row_ptr_) -> void {
            tile_desc.resize(tile_count);
            tile_desc_offset_ptr.resize(tile_ptr.size());

            set_bit_flag(row_ptr_);
            set_tile_descriptor_y_and_segsum_offsets();

            // we have some empty segments.
            if (tile_desc_offset_ptr.back())
                generate_tile_offset(row_ptr_);
        }

        auto generate_tile_pointer(const size_t nnz, std::span<const UnsignedType> row_ptr_) {
            tile_ptr.resize(tile_count + 1);

#pragma omp parallel for
            for (size_t tile_id = 0; tile_id < tile_ptr.size(); tile_id++) {
                // compute partition boundaries by partition of size sigma * omega,
                // clamp to [0, nnz]
                const auto boundary = static_cast<SignedType>(std::min(tile_id * sigma * omega, nnz));

                tile_ptr[tile_id] = {static_cast<UnsignedType>(detail::upper_bound_idx(row_ptr_, boundary))};
            }

            iterate_tiles([&](auto tile_id, auto start, auto stop) {
                if (start == stop)
                    return;

                // +1 because it needs to be inclusive.
                if (detail::is_dirty(row_ptr_.subspan(start, stop - start + 1)))
                    tile_ptr[tile_id].mark_dirty();
            });
        }

    public:
        static auto from_csr(const csr<ValueType, StorageContainer>& csr) -> csr5_info_t {
            const auto tile_size = sigma * omega;
            const auto num_non_zero = csr.values.size();

            const auto tile_count
              = static_cast<size_t>(std::floor(static_cast<double>(num_non_zero) / static_cast<double>(tile_size)));

            csr5_info_t result{.tile_count = tile_count};
            result.generate_tile_pointer(csr.values.size(), csr.row_start_offsets);
            result.generate_tile_descriptor(csr.row_start_offsets);

            return result;
        }
    };

    // same as CSR
    dimensions_t dimensions;
    StorageContainer<ValueType> vals;
    StorageContainer<UnsignedType> col_idx;
    StorageContainer<UnsignedType> row_ptr;

    csr5_info_t csr5_info;

    // TODO: these are irrelevant for the matrix itself, needed only for SPmV, make them arguments.
    size_t val_offset{0};
    bool skip_tail{false};

private:
    auto iterate_tiles(auto&&... args) const { csr5_info.iterate_tiles(std::forward<decltype(args)>(args)...); }

    auto transpose(std::ranges::random_access_range auto& to_transpose) -> void {
        using value_t = std::ranges::range_value_t<decltype(to_transpose)>;

        iterate_tiles([&](const auto tile_id, const auto row_start, const auto row_stop) {
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

            DIM_UNROLL
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

public:
    static auto from_csr(csr<ValueType, StorageContainer> csr) -> csr5 {
        auto&& info = csr5_info_t::from_csr(csr);

        csr5 val{.dimensions = csr.dimensions,
                 .vals = std::move(csr.values),
                 .col_idx = std::move(csr.col_indices),
                 .row_ptr = std::move(csr.row_start_offsets),
                 .csr5_info = std::move(info)};

        val.transpose();

        return val;
    }

    [[nodiscard]] auto tail_partition_start() const noexcept -> UnsignedType { return csr5_info.tile_ptr.back().idx(); }

    static auto load_x(dim::span<const ValueType> x, dim::span<const UnsignedType> column_index_partition,
                       size_t offset) noexcept {
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
        size_t tid{};
        UnsignedType start_row_start{};

        // these need to be aligned to 32 as the intrinsics require it.
        alignas(32) ValueType sum[8] = {};
        alignas(32) ValueType first_sum[8] = {};
        alignas(32) uint64_t cond[8] = {};
        alignas(32) int y_idx[16] = {};

        auto store(__m128i y_index, __m256d lsum, __m256i lcond) {
            _mm_store_si128(reinterpret_cast<__m128i*>(std::data(y_idx)), y_index);
            _mm256_store_pd(std::data(sum), lsum);
            _mm256_store_si256(reinterpret_cast<__m256i*>(std::data(cond)), lcond);
        }

        auto maybe_store(dim::span<ValueType> y, size_t idx) const noexcept {
            if (!cond[idx])
                return 0;

            y[static_cast<size_t>(y_idx[idx])] = sum[idx];
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
            else
                data.y[row_start] += val; // not the first to access this in our chunk.
        }

        struct vals_and_x
        {
            __m256d vals;
            __m256d x;
        };

        struct spmv_iteration_data
        {
            const spmv_thread_data& thread_data;
            size_t tile_id;
            dim::span<const ValueType> vals;
            dim::span<const UnsignedType> col_idx;

            auto load_vals_and_x(size_t offset) const noexcept -> vals_and_x {
                offset *= omega;

                return vals_and_x{
                  .vals = _mm256_load_pd(&vals[offset]),
                  .x = _mm256_set_pd(thread_data.data.x[col_idx[offset + 3]], thread_data.data.x[col_idx[offset + 2]],
                                     thread_data.data.x[col_idx[offset + 1]], thread_data.data.x[col_idx[offset]])};
            }

            auto partition_fast_track() const noexcept -> ValueType {
                auto sum256d = _mm256_setzero_pd();

                // fmadd of a single tile where all elements are from a same row.
                DIM_UNROLL
                for (size_t i = 0; i < sigma; i++) {
                    auto curr = load_vals_and_x(i);
                    sum256d = _mm256_fmadd_pd(curr.vals, curr.x, sum256d);
                }

                // total sum for this tile.
                return simd::hsum_avx(sum256d);
            }
        };

        auto get_iter_data(const csr5& self, size_t tile_id) const noexcept -> spmv_iteration_data {
            return spmv_iteration_data{.thread_data = *this,
                                       .tile_id = tile_id,
                                       .vals = dim::span{self.vals}.subspan(tile_id * tile_size(), tile_size()),
                                       .col_idx = dim::span{self.col_idx}.subspan(tile_id * tile_size(), tile_size())};
        }
    };

    enum class spmv_strategy
    {
        //! @brief adresses output vector directly.
        absolute,
        //! @brief adresses output vector relatively to where first tile starts.
        partial
    };

    template<spmv_strategy Strategy = spmv_strategy::absolute>
    [[nodiscard]] auto tile_ptr_accessor() const noexcept {
        if constexpr (Strategy == spmv_strategy::absolute) {
            return [this](size_t idx) { return csr5_info.tile_ptr[idx]; };
        } else {
            return [tiles = std::span{csr5_info.tile_ptr}, row_offset = csr5_info.tile_ptr.front().idx()](size_t idx) {
                return tiles[idx].relative(row_offset);
            };
        }
    }

    struct spmv_parallel_data_t
    {
        size_t thread_count{static_cast<size_t>(::omp_get_max_threads())};
        size_t chunk_size{};
        size_t active_thread_count{};

        [[nodiscard]] auto is_active(size_t tid) const noexcept -> bool { return tid < active_thread_count; }
        [[nodiscard]] auto calibrator_count() const noexcept -> size_t {
            return std::min(thread_count, active_thread_count);
        }
    };

    [[nodiscard]] auto get_parallel_data() const noexcept -> spmv_parallel_data_t {
        const auto thread_count = static_cast<size_t>(::omp_get_max_threads());
        const auto chunk_size = static_cast<size_t>(
          std::ceil(static_cast<double>(csr5_info.tile_count) / static_cast<double>(thread_count)));

        return {.thread_count = thread_count,
                .chunk_size = chunk_size,
                .active_thread_count = static_cast<size_t>(
                  std::ceil(static_cast<double>(csr5_info.tile_count) / static_cast<double>(chunk_size)))};
    }

    auto spmv_full_partitions(spmv_data_t spmv_data, const spmv_parallel_data_t& parallel_data, auto&& accessor) const {
#pragma omp parallel
        {
            const auto tid = static_cast<size_t>(::omp_get_thread_num());

            spmv_thread_data thread_data{
              .data = spmv_data,
              .tid = tid,
              .start_row_start = parallel_data.is_active(tid) ? accessor(tid * parallel_data.chunk_size).idx() : 0};

#pragma omp for schedule(static, parallel_data.chunk_size)
            for (size_t tile_id = 0; tile_id < csr5_info.tile_count; ++tile_id) {
                const auto iteration_data = thread_data.get_iter_data(*this, tile_id);

                const auto current_ptr = accessor(tile_id);
                const auto row_stop = accessor(tile_id + 1).idx();

                const auto current_descriptor = csr5_info.tile_desc[tile_id];
                const auto fast_direct = has_rbit_set(current_descriptor.columns[0].bit_flag, 0);

                // the tile only has elements from a single row.
                if (const auto row_start = current_ptr.raw; row_start == row_stop) {
                    thread_data.store_to_row_start(row_start, iteration_data.partition_fast_track(), fast_direct);
                    continue;
                }

                const auto empty_rows = current_ptr.is_dirty();
                const auto row_start = current_ptr.idx();

                auto y_local = spmv_data.y.subspan(row_start);

                const auto offset_pointer = empty_rows ? csr5_info.empty_offset_for(tile_id) : 0;
                const auto compute_y_idx = [&, empty_rows, offset_pointer](auto&& offset) {
                    return empty_rows ? _mm_i32gather_epi32(
                             reinterpret_cast<const int*>(&csr5_info.tile_desc_offset[offset_pointer]), offset, 4)
                                      : offset;
                };

                auto vec = current_descriptor.vectorized();

                // remember if the first element of this partition is the first element of a new row
                auto local_bit256i = vec.get_local_bit(0);
                _mm256_store_si256(reinterpret_cast<__m256i*>(std::data(thread_data.cond)), local_bit256i);

                // remember if the first element of the first partition of the current thread is the first element of a
                // new row
                const bool first_all_direct = tile_id == tid * parallel_data.chunk_size && fast_direct;

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
                auto curr = iteration_data.load_vals_and_x(0);
                auto sum256d = _mm256_mul_pd(curr.vals, curr.x);

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
                    curr = iteration_data.load_vals_and_x(i);
                    sum256d = _mm256_fmadd_pd(curr.vals, curr.x, sum256d);
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

    auto spmv_calibrator(spmv_data_t spmv_data, const spmv_parallel_data_t& parallel_data,
                         auto&& accessor) const noexcept {
        constexpr auto stride = spmv_thread_data::stride;

        for (size_t i = 0; i < parallel_data.calibrator_count(); i++)
            spmv_data.y[accessor(i * parallel_data.chunk_size).idx()] += spmv_data.calibrator[i * stride];
    }

    template<spmv_strategy Strategy = spmv_strategy::absolute>
    auto spmv_tail_partition(spmv_data_t spmv_data) const noexcept {
        if (skip_tail)
            return;

        // TODO: abstract this away to a separate method as well.
        const auto offset = Strategy == spmv_strategy::absolute ? 0 : csr5_info.tile_ptr.front().idx();

        const auto first_tail_element = csr5_info.tile_count * tile_size();

        const auto tail_start = tail_partition_start() - offset;
        const auto tail_stop = dimensions.rows - offset;

#pragma omp parallel for
        for (size_t row = tail_start; row < tail_stop; ++row) {
            auto idx_start = row == tail_start ? first_tail_element : (row_ptr[row] - val_offset);
            auto idx_stop = row_ptr[row + 1] - val_offset;

            ValueType sum = 0;
            for (auto idx = idx_start; idx < idx_stop; ++idx)
                sum += vals[idx] * spmv_data.x[col_idx[idx]];

            // row started in some tile before.
            if (row == tail_start && row_ptr[row] != first_tail_element)
                spmv_data.y[row] += sum;
            else
                spmv_data.y[row] = sum;
        }
    }

    template<spmv_strategy Strategy = spmv_strategy::absolute>
    auto spmv(spmv_data_t spmv_data) const {
        if (spmv_data.x.size() != dimensions.cols)
            throw std::invalid_argument{"can't multiply, vector size doesn't match column count of the matrix"};

        const auto parallel_data = get_parallel_data();
        const auto accessor = tile_ptr_accessor<Strategy>();

        spmv_full_partitions(spmv_data, parallel_data, accessor);
        spmv_calibrator(spmv_data, parallel_data, accessor);
        spmv_tail_partition<Strategy>(spmv_data);
    }

    template<spmv_strategy Strategy = spmv_strategy::absolute>
    auto spmv(dim::span<const ValueType> x, dim::span<ValueType> y, dim::span<ValueType> calibrator) const {
        spmv<Strategy>(spmv_data_t{.x = x, .y = y, .calibrator = calibrator});
    }

    auto allocate_calibrator() const noexcept -> StorageContainer<ValueType> {
        return StorageContainer<ValueType>(static_cast<size_t>(::omp_get_max_threads())
                                             * dim::memory::hardware_destructive_interference_size / sizeof(ValueType),
                                           ValueType{});
    }

    template<spmv_strategy Strategy = spmv_strategy::absolute>
    auto spmv(dim::span<const ValueType> x, dim::span<ValueType> y) const {
        auto calibrator = allocate_calibrator();
        return spmv<Strategy>(x, y, calibrator);
    }

    [[nodiscard]] auto first_tile_uncapped() const noexcept -> bool {
        return csr5_info.tile_desc.front().is_uncapped();
    }

    [[nodiscard]] auto first_row_idx() const noexcept { return csr5_info.first_row_idx(); }

    [[nodiscard]] auto last_row_idx() const noexcept {
        return skip_tail ? csr5_info.last_row_idx() : dimensions.rows - 1;
    }
};

} // namespace dim::mat

#endif /* INCLUDE_DIM_MAT_STORAGE_FORMATS_CSR5 */
