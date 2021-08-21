#ifndef THIRD_PARTY_CSR5_INCLUDE_DETAIL_AVX2_FORMAT_AVX2
#define THIRD_PARTY_CSR5_INCLUDE_DETAIL_AVX2_FORMAT_AVX2

#include "anonymouslib_avx2.h"
#include "common_avx2.h"
#include "detail/utils.h"
#include "utils_avx2.h"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <type_traits>
#include <utility>

#include <dim/span.h>

#include <dim/memory/aligned_allocator.h>

namespace csr5::avx2 {

// TODO: move these helpers to a different header and add unit-tests.

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

template<typename Ty>
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
constexpr size_t set_bit_count(const Integral value, const size_t start = 0, const size_t count = bit_size<Integral>) {
    size_t result{};

#pragma unroll
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

template<typename iT, typename uiT>
void generate_partition_pointer_s1_kernel(dim::span<const iT> row_start_offsets, const dim::span<uiT> partition,
                                          const size_t sigma, const size_t nnz) {
#pragma omp parallel for
    for (size_t global_id = 0; global_id < partition.size(); global_id++) {
        // compute partition boundaries by partition of size sigma * omega,
        // clamp to [0, nnz]
        const auto boundary = static_cast<iT>(std::min(global_id * sigma * ANONYMOUSLIB_CSR5_OMEGA, nnz));

        partition[global_id] = static_cast<uiT>(upper_bound_idx(row_start_offsets, boundary));
    }
}

template<typename iT, typename uiT>
void generate_partition_pointer_s2_kernel(const dim::span<const iT> row_start_offsets, const dim::span<uiT> partition) {
    iterate_partitions(partition, [&](auto partition_id, auto start, auto stop) {
        if (start == stop)
            return;

        if (is_dirty(row_start_offsets.subspan(start, stop - start)))
            partition[partition_id] = mark_dirty(start);
    });
}

template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT>
int generate_partition_pointer(const size_t sigma, const size_t num_non_zero,
                               const dim::span<ANONYMOUSLIB_UIT> partition,
                               const dim::span<const ANONYMOUSLIB_IT> row_start_offsets) {
    // step 1. binary search row pointer
    generate_partition_pointer_s1_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>(row_start_offsets, partition, sigma,
                                                                            num_non_zero);

    // step 2. check empty rows
    generate_partition_pointer_s2_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>(row_start_offsets, partition);

    return ANONYMOUSLIB_SUCCESS;
}

/**
 * @brief Sets the bit flag of each first non-0 element in a tile to true.
 */
template<typename iT, typename uiT>
void set_partition_descriptor_bit_flags(const iT* row_start_offsets, const dim::span<const uiT> partition_pointer,
                                        uiT* partition_descriptor, const size_t sigma, const size_t bit_all_offset,
                                        const size_t num_packet) {
    iterate_partitions(partition_pointer, [&](auto partition_id, auto row_start, auto row_stop) {
        // start of tiles for this partition.
        const auto location_base = partition_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet;

        for (auto rid = row_start; rid <= row_stop; rid++) {
            const auto idx = static_cast<size_t>(row_start_offsets[rid]);
            // if we aren't in the correct partition, skip.
            if (partition_id != idx / (ANONYMOUSLIB_CSR5_OMEGA * sigma))
                continue;

            // the tile has sigma rows and omega cols, thus we get row by doing
            // modulo sigma.
            const auto tile_row = idx % sigma;
            // each element in tile gets a bit flag, offset by
            // bit_all_offset(y_offset + seg_offset).
            const auto tile_bit_flag_offset = tile_row + bit_all_offset;
            // get in which row to store the bit flag.
            const auto tile_y = tile_bit_flag_offset / bit_size<uiT>;
            // get the column in which to store the bit flag.
            const auto tile_x = (idx / sigma) % ANONYMOUSLIB_CSR5_OMEGA;
            // get the index in the partition_descriptor storage.
            const auto location = location_base + tile_y * ANONYMOUSLIB_CSR5_OMEGA + tile_x;

            // set the bit flag as this is the first non-0 element in the row in
            // this tile.
            partition_descriptor[location] |= set_bit<uiT>(tile_bit_flag_offset % bit_size<uiT>);
        }
    });
}

template<typename ReadPacketFn>
auto num_set_bits_in_bitflag(size_t bit_all_offset, size_t sigma, size_t col_idx, ReadPacketFn&& read_packet) {
    using packet_type = std::remove_cv_t<std::remove_reference_t<decltype(read_packet(0, 0))>>;
    const auto first_packet_bit_flag_size = bit_size<packet_type> - bit_all_offset;

    struct
    {
        bool first_set{false};
        size_t num_set{0};
    } result;

    auto packet = read_packet(0, col_idx);

    // check if bit is set or col_idx is 0 in which case the big flag is always
    // true.
    result.first_set = !col_idx || has_bit_set(packet, bit_all_offset);
    result.num_set = result.first_set;

    // process the values in the first packet.
    const auto processed = std::min(first_packet_bit_flag_size, sigma);
    // process one less than the count because we've processed the first bit to
    // compute start.
    result.num_set += set_bit_count(packet, bit_all_offset + 1, processed - 1);

    for (auto remaining = sigma - processed, row_idx = size_t{1}; remaining != 0; ++row_idx) {
        packet = read_packet(row_idx, col_idx);

        // compute how much we still need to sum.
        const auto loop_processed = std::min(remaining, bit_size<decltype(packet)>);
        result.num_set += set_bit_count(packet, 0, loop_processed);

        // subtract the processed data from remaining data.
        remaining -= loop_processed;
    }

    return result;
}

template<typename ReadPacket>
void calculate_segn_scan_and_present(const size_t sigma, const size_t bit_all_offset, const dim::span<int> segn_scan,
                                     const dim::span<bool> present, const ReadPacket& read_packet) {
#pragma omp simd
    for (size_t col_idx = 0; col_idx < ANONYMOUSLIB_CSR5_OMEGA; ++col_idx) {
        const auto [first_set, num_set] = num_set_bits_in_bitflag(bit_all_offset, sigma, col_idx, read_packet);

        const bool current_present = num_set != 0;

        segn_scan[col_idx] = std::max(static_cast<int>(num_set) - 1 + current_present, 0);
        present[col_idx] = current_present;
    }

    std::exclusive_scan(segn_scan.begin(), segn_scan.end(), segn_scan.begin(), 0);
}

template<typename uiT>
auto create_packet_accessor(uiT* partition_descriptor, size_t partition_id, size_t num_packets) {
    const auto base_descriptor_index = partition_id * ANONYMOUSLIB_CSR5_OMEGA * num_packets;

    return [partition_descriptor, base_descriptor_index](size_t row, size_t col) -> uiT& {
        return partition_descriptor[base_descriptor_index + row * ANONYMOUSLIB_CSR5_OMEGA + col];
    };
}

template<typename iT, typename uiT>
void set_partition_descriptor_y_and_segsum_offsets(const dim::span<const uiT> partitions, uiT* partition_descriptor,
                                                   iT* partition_descriptor_offset_pointer, const size_t sigma,
                                                   const size_t num_packet, const size_t bit_y_offset,
                                                   const size_t bit_scansum_offset) {
    const auto bit_all_offset = bit_y_offset + bit_scansum_offset;

    iterate_partitions(partitions, [&](auto par_id, auto row_start, auto row_stop) {
        // skip empty rows.
        if (row_start == row_stop)
            return;

        int segn_scan[ANONYMOUSLIB_CSR5_OMEGA + 1] = {};
        bool present[ANONYMOUSLIB_CSR5_OMEGA] = {};

        const auto access_packet = create_packet_accessor(partition_descriptor, par_id, num_packet);
        // partition before was dirty.
        const auto has_empty_rows = is_dirty(partitions[par_id]);
        calculate_segn_scan_and_present(sigma, bit_all_offset, segn_scan, present, access_packet);

        if (has_empty_rows) {
            partition_descriptor_offset_pointer[par_id] = segn_scan[ANONYMOUSLIB_CSR5_OMEGA];
            partition_descriptor_offset_pointer[partitions.size() - 1] += segn_scan[ANONYMOUSLIB_CSR5_OMEGA];
        }

        const auto base_descriptor_index = par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet;

#pragma omp simd
        for (size_t col_idx = 0; col_idx < ANONYMOUSLIB_CSR5_OMEGA; col_idx++) {
            auto first_packet = access_packet(0, col_idx);

            const auto y_offset = col_idx ? static_cast<uiT>(segn_scan[col_idx]) - 1 : 0;
            first_packet |= y_offset << (bit_size<decltype(first_packet)> - bit_y_offset);

            const auto scansum_offset
              = present[col_idx]
                  ? static_cast<uiT>(count_consecutive_equal_elements(dim::span{present}.subspan(col_idx + 1), false))
                  : 0;
            first_packet |= scansum_offset << (bit_size<decltype(first_packet)> - bit_all_offset);

            access_packet(0, col_idx) = partition_descriptor[base_descriptor_index + col_idx] = first_packet;
        }
    });
}

template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT>
int generate_partition_descriptor(const size_t sigma, const size_t bit_y_offset, const size_t bit_scansum_offset,
                                  const size_t num_packet, const ANONYMOUSLIB_IT* row_pointer,
                                  const dim::span<const ANONYMOUSLIB_UIT> partition_pointer,
                                  ANONYMOUSLIB_UIT* partition_descriptor,
                                  ANONYMOUSLIB_IT* partition_descriptor_offset_pointer, ANONYMOUSLIB_IT& _num_offsets) {
    size_t bit_all_offset = bit_y_offset + bit_scansum_offset;

    set_partition_descriptor_bit_flags<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>(
      row_pointer, partition_pointer, partition_descriptor, sigma, bit_all_offset, num_packet);

    set_partition_descriptor_y_and_segsum_offsets<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>(
      partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, sigma, num_packet, bit_y_offset,
      bit_scansum_offset);

    // if we have any empty rows, this will be non-0 and we have to adjust the
    // offsets.
    if (partition_descriptor_offset_pointer[partition_pointer.size() - 1])
        std::exclusive_scan(partition_descriptor_offset_pointer,
                            partition_descriptor_offset_pointer + partition_pointer.size(),
                            partition_descriptor_offset_pointer, 0);

    _num_offsets = partition_descriptor_offset_pointer[partition_pointer.size() - 1];

    return ANONYMOUSLIB_SUCCESS;
}

template<typename iT, typename uiT>
void generate_partition_descriptor_offset_kernel(dim::span<const iT> rows, const dim::span<const uiT> partitions,
                                                 const uiT* partition_descriptor,
                                                 const iT* partition_descriptor_offset_pointer,
                                                 iT* partition_descriptor_offset, const size_t num_packet,
                                                 const size_t bit_y_offset, const size_t bit_scansum_offset,
                                                 const size_t sigma) {
    const size_t bit_all_offset = bit_y_offset + bit_scansum_offset;

    iterate_partitions(partitions, [&](auto par_id, auto row_start, auto row_stop) {
        if (!is_dirty(partitions[par_id]))
            return;

        const auto offset_pointer = static_cast<size_t>(partition_descriptor_offset_pointer[par_id]);

#pragma omp simd
        for (size_t col_idx = 0; col_idx < ANONYMOUSLIB_CSR5_OMEGA; col_idx++) {
            const auto access_packet = create_packet_accessor(partition_descriptor, par_id, num_packet);

            auto y_offset = access_packet(0, col_idx) >> (bit_size<uiT> - bit_y_offset);
            const auto [_, num_set] = num_set_bits_in_bitflag(bit_all_offset, sigma, col_idx, access_packet);

            for (size_t i = 0; i < num_set; ++i, ++y_offset) {
                const auto row_data = rows.subspan(row_start + 1, row_stop - row_start);
                const auto idx = static_cast<iT>(par_id * ANONYMOUSLIB_CSR5_OMEGA * sigma + col_idx * sigma);

                partition_descriptor_offset[offset_pointer + y_offset]
                  = static_cast<iT>(upper_bound_idx(row_data, idx));
            }
        }
    });
}

template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT>
int generate_partition_descriptor_offset(const size_t sigma, const size_t bit_y_offset, const size_t bit_scansum_offset,
                                         const size_t num_packet, dim::span<const ANONYMOUSLIB_IT> rows,
                                         const dim::span<const ANONYMOUSLIB_UIT> partition_pointer,
                                         ANONYMOUSLIB_UIT* partition_descriptor,
                                         ANONYMOUSLIB_IT* partition_descriptor_offset_pointer,
                                         ANONYMOUSLIB_IT* partition_descriptor_offset) {
    generate_partition_descriptor_offset_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>(
      rows, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset,
      num_packet, bit_y_offset, bit_scansum_offset, sigma);

    return ANONYMOUSLIB_SUCCESS;
}

// R2C==true means CSR->CSR5, otherwise CSR5->CSR
template<bool R2C, typename T, typename uiT>
void aosoa_transpose_kernel_smem(T* d_data, const dim::span<uiT> partitions, const size_t sigma) {
    const auto transform_index = [&](size_t idx, bool r2c) {
        if (r2c)
            return std::pair{(idx / sigma), (idx % sigma)};

        return std::pair{(idx % ANONYMOUSLIB_CSR5_OMEGA), (idx / ANONYMOUSLIB_CSR5_OMEGA)};
    };

    const size_t partition_size = sigma * ANONYMOUSLIB_CSR5_OMEGA;
    dim::memory::cache_aligned_vector<T> s_data_all(partition_size * static_cast<size_t>(omp_get_max_threads()));

    // TODO: figure out why it doesn't like the count of partitions here.
    iterate_partitions(partitions.subspan(0, partitions.size() - 1), [&](auto par_id, auto start, auto stop) {
        T* s_data = &s_data_all[partition_size * static_cast<size_t>(omp_get_thread_num())];

        // if this is fast track partition, do not transpose it
        if (start == stop)
            return;

#pragma omp simd
        for (size_t idx = 0; idx < ANONYMOUSLIB_CSR5_OMEGA * sigma; idx++) {
            const auto [x, y] = transform_index(idx, R2C);
            s_data[y * ANONYMOUSLIB_CSR5_OMEGA + x] = d_data[par_id * ANONYMOUSLIB_CSR5_OMEGA * sigma + idx];
        }

// store transposed shared mem data to global
#pragma omp simd
        for (size_t idx = 0; idx < ANONYMOUSLIB_CSR5_OMEGA * sigma; idx++) {
            const auto [x, y] = transform_index(idx, !R2C);
            d_data[par_id * ANONYMOUSLIB_CSR5_OMEGA * sigma + idx] = s_data[y * ANONYMOUSLIB_CSR5_OMEGA + x];
        }
    });
}

template<bool R2C, typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT, typename ANONYMOUSLIB_VT>
int aosoa_transpose(const size_t sigma, const dim::span<ANONYMOUSLIB_UIT> partition_pointer,
                    ANONYMOUSLIB_IT* column_index, ANONYMOUSLIB_VT* value) {
    aosoa_transpose_kernel_smem<R2C, ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>(column_index, partition_pointer, sigma);
    aosoa_transpose_kernel_smem<R2C, ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT>(value, partition_pointer, sigma);

    return ANONYMOUSLIB_SUCCESS;
}

} // namespace csr5::avx2

#endif /* THIRD_PARTY_CSR5_INCLUDE_DETAIL_AVX2_FORMAT_AVX2 */
