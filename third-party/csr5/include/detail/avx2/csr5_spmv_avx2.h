#ifndef THIRD_PARTY_CSR5_INCLUDE_DETAIL_AVX2_CSR5_SPMV_AVX2
#define THIRD_PARTY_CSR5_INCLUDE_DETAIL_AVX2_CSR5_SPMV_AVX2

#include "common_avx2.h"
#include "detail/avx2/format_avx2.h"
#include "utils_avx2.h"

#include <bit>
#include <immintrin.h>

namespace csr5::avx2 {

template<typename vT, typename iT>
auto load_x(const vT* x, iT* column_index_partition, size_t offset) {
    return _mm256_set_pd(x[column_index_partition[offset + 3]], x[column_index_partition[offset + 2]],
                         x[column_index_partition[offset + 1]], x[column_index_partition[offset]]);
}

template<typename iT, typename vT>
inline void partition_fast_track(const vT* value_partition, const vT* x, const iT* column_index_partition,
                                 vT* calibrator, vT* y, const iT row_start, const int tid, const iT start_row_start,
                                 const int stride_vT, const bool direct) {
    auto sum256d = _mm256_setzero_pd();

#pragma unroll
    for (size_t i = 0; i < ANONYMOUSLIB_CSR5_SIGMA; i++) {
        const auto base_offset = i * ANONYMOUSLIB_CSR5_OMEGA;

        sum256d = _mm256_fmadd_pd(_mm256_load_pd(&value_partition[base_offset]),
                                  load_x(x, column_index_partition, base_offset), sum256d);
    }

    vT sum = hsum_avx(sum256d);

    if (row_start == start_row_start && !direct)
        calibrator[tid * stride_vT] += sum;
    else {
        if (direct)
            y[row_start] = sum;
        else
            y[row_start] += sum;
    }
}

template<typename iT>
auto load_partition_info(iT* partition_descriptor, int bit_y_offset, int bit_scansum_offset) {
    const auto* partition_descriptor128i = reinterpret_cast<const __m128i*>(partition_descriptor);
    auto descriptor128i = _mm_load_si128(partition_descriptor128i);

    auto y_offset128i = _mm_srli_epi32(descriptor128i, 32 - bit_y_offset);
    auto scansum_offset128i = _mm_srli_epi32(_mm_slli_epi32(descriptor128i, bit_y_offset), 32 - bit_scansum_offset);
    descriptor128i = _mm_slli_epi32(descriptor128i, bit_y_offset + bit_scansum_offset);

    return std::tuple{descriptor128i, y_offset128i, scansum_offset128i};
}

inline __m256i get_local_bit(__m128i descriptor, int offset = 0) {
    return _mm256_and_si256(_mm256_cvtepu32_epi64(_mm_srli_epi32(descriptor, 31 - offset)), _mm256_set1_epi64x(0x1));
}

inline __m256d shuffle_by_scansum(__m256d sum, __m128i scansum) {
    // add base relative offsets, cast to 256 bits.
    auto shuffle_mask = _mm256_castsi128_si256(_mm_add_epi32(scansum, _mm_set_epi32(3, 2, 1, 0)));
    // copy each 32bit part from the lower 128 bits to its neighbour.
    shuffle_mask = _mm256_permutevar8x32_epi32(shuffle_mask, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
    // x2 as we're shuffling 32-bit halves but aliasing 64-bit doubles (e.g 1st double is at index 2 etc.).
    shuffle_mask = _mm256_add_epi32(shuffle_mask, shuffle_mask);
    // +1 to every second to have "upper" and "lower" parts of the doubles.
    shuffle_mask = _mm256_add_epi32(shuffle_mask, _mm256_set_epi32(1, 0, 1, 0, 1, 0, 1, 0));
    // shuffle by the created mask.
    return _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(sum), shuffle_mask));
}

template<typename iT, typename uiT, typename vT>
void spmv_csr5_compute_kernel(const iT* column_index, const vT* value, const vT* x, const uiT* partition_pointer,
                              const uiT* partition_descriptor, const iT* partition_descriptor_offset_pointer,
                              const iT* partition_descriptor_offset, vT* calibrator, vT* y, const iT p,
                              const int num_packet, const int bit_y_offset, const int bit_scansum_offset,
                              const int c_sigma) {
    const int num_thread = omp_get_max_threads();
    const auto chunk = static_cast<int>(std::ceil(static_cast<double>(p - 1) / static_cast<double>(num_thread)));
    const int stride_vT = ANONYMOUSLIB_X86_CACHELINE / sizeof(vT);
    const auto num_thread_active = static_cast<int>(std::ceil((p - 1.0) / chunk));

#pragma omp parallel
    {
        const auto tid = omp_get_thread_num();
        iT start_row_start = tid < num_thread_active ? strip_dirty(partition_pointer[tid * chunk]) : 0;

        // these need to be aligned to 32 as the intrinsics require it.
        alignas(32) vT s_sum[8];        // allocate a cache line
        alignas(32) vT s_first_sum[8];  // allocate a cache line
        alignas(32) uint64_t s_cond[8]; // allocate a cache line
        alignas(32) int s_y_idx[16];    // allocate a cache line

#pragma omp for schedule(static, chunk)
        for (int par_id = 0; par_id < p - 1; par_id++) {
            const auto partition_offset_base = par_id * ANONYMOUSLIB_CSR5_OMEGA;
            const auto partition_descriptor_base = partition_offset_base * num_packet;

            // column indices for this partition.
            const iT* column_index_partition = &column_index[partition_offset_base * c_sigma];
            // values for this partition.
            const vT* value_partition = &value[partition_offset_base * c_sigma];

            // TODO: check if the signedness used is needed.
            auto row_start = static_cast<iT>(partition_pointer[par_id]);
            const iT row_stop = strip_dirty(partition_pointer[par_id + 1]);

            if (row_start == row_stop) {
                // fast track through reduction
                // check whether the the partition contains the first element of row "row_start"
                // => we are the first writing data to d_y[row_start]
                bool fast_direct
                  = has_bit_set(partition_descriptor[partition_descriptor_base], bit_y_offset + bit_scansum_offset);
                partition_fast_track<iT, vT>(value_partition, x, column_index_partition, calibrator, y, row_start, tid,
                                             start_row_start, stride_vT, fast_direct);

            } else {
                // normal track for all the other partitions
                const bool empty_rows = is_dirty(row_start);
                row_start = strip_dirty(row_start);

                vT* y_local = &y[row_start + 1];
                const int offset_pointer = empty_rows ? partition_descriptor_offset_pointer[par_id] : 0;

                auto [descriptor128i, y_offset128i, scansum_offset128i] = load_partition_info(
                  &partition_descriptor[partition_descriptor_base], bit_y_offset, bit_scansum_offset);

                // remember if the first element of this partition is the first element of a new row
                auto local_bit256i = get_local_bit(descriptor128i, 0);
                _mm256_store_si256(reinterpret_cast<__m256i*>(std::data(s_cond)), local_bit256i);
                const bool first_direct = s_cond[0] != 0;

                // remember if the first element of the first partition of the current thread is the first element of a
                // new row
                bool first_all_direct = par_id == tid * chunk && first_direct;

                // set the 0th bit of the descriptor to 1.
                descriptor128i = _mm_or_si128(descriptor128i, _mm_set_epi32(0, 0, 0, std::bit_cast<int>(0x80000000)));

                // load bits for next 4 reductions.
                local_bit256i = get_local_bit(descriptor128i, 0);

                // start256i = !local_bit256i (speaking in bools). Meaning it's true if no row started here.
                auto start256i = _mm256_sub_epi64(_mm256_set1_epi64x(0x1), local_bit256i);

                auto stop256i = _mm256_setzero_si256();
                auto direct256i = _mm256_and_si256(local_bit256i, _mm256_set_epi64x(0x1, 0x1, 0x1, 0));

                // do the first sum.
                auto value256d = _mm256_load_pd(value_partition);
                auto x256d = load_x(x, column_index_partition, 0);
                auto sum256d = _mm256_mul_pd(value256d, x256d);

                // step 1. thread-level seg sum
#if ANONYMOUSLIB_CSR5_SIGMA > 23
                int ly = 0;
#endif
                auto first_sum256d = _mm256_setzero_pd();
                for (int i = 1; i < ANONYMOUSLIB_CSR5_SIGMA; i++) {
#if ANONYMOUSLIB_CSR5_SIGMA > 23
                    int norm_i = i - (32 - bit_y_offset - bit_scansum_offset);

                    if (!(ly || norm_i) || (ly && !(norm_i % 32))) {
                        ly++;
                        descriptor128i = _mm_load_si128(&partition_descriptor128i[ly]);
                    }
                    norm_i = !ly ? i : norm_i;
                    norm_i = 31 - norm_i % 32;

                    local_bit256i = _mm256_and_si256(_mm256_cvtepu32_epi64(_mm_srli_epi32(descriptor128i, norm_i)),
                                                     _mm256_set1_epi64x(0x1));
#else
                    local_bit256i = get_local_bit(descriptor128i, i);
#endif

                    constexpr auto all_set_mask = std::bit_cast<int64_t>(0xFFFFFFFFFFFFFFFF);
                    int store_to_offchip = _mm256_testz_si256(local_bit256i, _mm256_set1_epi64x(all_set_mask));

                    // any of the bits mark start of a new row.
                    if (!store_to_offchip) {
                        // if empty rows we need to use empty_offset[y_offset]
                        auto y_idx128i = empty_rows ? _mm_i32gather_epi32(&partition_descriptor_offset[offset_pointer],
                                                                          y_offset128i, 4)
                                                    : y_offset128i;

                        // mask scatter store
                        _mm_store_si128(reinterpret_cast<__m128i*>(std::data(s_y_idx)), y_idx128i);
                        _mm256_store_pd(s_sum, sum256d);

                        // s_cond means we have a new row started here (direct256i keeps track if any row was started),
                        // thus we need to store the partial sum we've done to
                        _mm256_store_si256(reinterpret_cast<__m256i*>(std::data(s_cond)),
                                           _mm256_and_si256(direct256i, local_bit256i));
                        const auto store_if_new_row = [&s_cond, &y_local, &s_y_idx, &s_sum](auto idx) {
                            if (!s_cond[idx])
                                return 0;
                            y_local[s_y_idx[idx]] = s_sum[idx];
                            return 1;
                        };

                        // if any of the lanes stored, we need to increase its index (new row == new output in Y).
                        y_offset128i
                          = _mm_add_epi32(y_offset128i, _mm_set_epi32(store_if_new_row(3), store_if_new_row(2),
                                                                      store_if_new_row(1), store_if_new_row(0)));

                        // -1 if direct==0 (AKA no row active yet) && local bit is set (AKA start of new row).
                        // end of red subsegment in CSR5 paper.
                        const auto localbit_mask = _mm256_cmpeq_epi64(local_bit256i, _mm256_set1_epi64x(0x1));
                        const auto mask
                          = _mm256_andnot_si256(_mm256_cmpeq_epi64(direct256i, _mm256_set1_epi64x(0x1)), localbit_mask);

                        // if mask is -1 (AKA uint64_t::max) we take values from first_sum256d, else we take from
                        // sum256d.
                        first_sum256d = _mm256_add_pd(_mm256_andnot_pd(_mm256_castsi256_pd(mask), first_sum256d),
                                                      _mm256_and_pd(_mm256_castsi256_pd(mask), sum256d));

                        // zero out the parts of sum which have localbit set (AKA starting new row).
                        sum256d = _mm256_andnot_pd(_mm256_castsi256_pd(localbit_mask), sum256d);

                        // set direct to know if we've started a row yet.
                        direct256i = _mm256_or_si256(direct256i, local_bit256i);
                        // TODO: understand.
                        stop256i = _mm256_add_epi64(stop256i, local_bit256i);
                    }

                    // process the next column in bitmap.
                    x256d = load_x(x, column_index_partition, i * ANONYMOUSLIB_CSR5_OMEGA);
                    value256d = _mm256_load_pd(&value_partition[i * ANONYMOUSLIB_CSR5_OMEGA]);
                    sum256d = _mm256_fmadd_pd(value256d, x256d, sum256d);
                }

                // check if any lane had a row start in there.
                auto tmp256i = _mm256_cmpeq_epi64(direct256i, _mm256_set1_epi64x(0x1));
                // null out the elements of first_sum256d which didn't have a row starting there.
                first_sum256d = _mm256_add_pd(_mm256_and_pd(_mm256_castsi256_pd(tmp256i), first_sum256d),
                                              _mm256_andnot_pd(_mm256_castsi256_pd(tmp256i), sum256d));

                auto last_sum256d = sum256d;
                // make a mask out of it.
                tmp256i = _mm256_cmpeq_epi64(start256i, _mm256_set1_epi64x(0x1));
                sum256d = _mm256_and_pd(_mm256_castsi256_pd(tmp256i), first_sum256d);

                sum256d = _mm256_permute4x64_pd(sum256d, make_permute_seq(1, 2, 3, 0));
                const auto zero_last_mask = _mm256_castsi256_pd(_mm256_set_epi64x(
                  0x0000000000000000, std::bit_cast<int64_t>(0xFFFFFFFFFFFFFFFF),
                  std::bit_cast<int64_t>(0xFFFFFFFFFFFFFFFF), std::bit_cast<int64_t>(0xFFFFFFFFFFFFFFFF)));
                sum256d = _mm256_and_pd(zero_last_mask, sum256d);

                const auto tmp_sum256d = sum256d;
                // inclusive prefix scan.
                sum256d = hscan_avx(sum256d);

                // in[i] = in[i + seg_offset[i]] - in[i]
                sum256d = _mm256_sub_pd(shuffle_by_scansum(sum256d, scansum_offset128i), sum256d);
                // in[i] -= tmp[i]
                sum256d = _mm256_add_pd(sum256d, tmp_sum256d);

                tmp256i = _mm256_cmpgt_epi64(start256i, stop256i);
                last_sum256d = _mm256_add_pd(last_sum256d, _mm256_andnot_pd(_mm256_castsi256_pd(tmp256i), sum256d));

                auto y_idx128i = empty_rows
                                   ? _mm_i32gather_epi32(&partition_descriptor_offset[offset_pointer], y_offset128i, 4)
                                   : y_offset128i;

                _mm256_store_si256(reinterpret_cast<__m256i*>(std::data(s_cond)), direct256i);
                _mm_store_si128(reinterpret_cast<__m128i*>(std::data(s_y_idx)), y_idx128i);
                _mm256_store_pd(s_sum, last_sum256d);

                if (s_cond[0]) {
                    y_local[s_y_idx[0]] = s_sum[0];
                    _mm256_store_pd(s_first_sum, first_sum256d);
                }
                if (s_cond[1])
                    y_local[s_y_idx[1]] = s_sum[1];
                if (s_cond[2])
                    y_local[s_y_idx[2]] = s_sum[2];
                if (s_cond[3])
                    y_local[s_y_idx[3]] = s_sum[3];

                // only use calibrator if this partition does not contain the first element of the row "row_start"
                if (row_start == start_row_start && !first_all_direct)
                    calibrator[tid * stride_vT] += s_cond[0] ? s_first_sum[0] : s_sum[0];
                else {
                    if (first_direct)
                        y[row_start] = s_cond[0] ? s_first_sum[0] : s_sum[0];
                    else
                        y[row_start] += s_cond[0] ? s_first_sum[0] : s_sum[0];
                }
            }
        }
    }
}

template<typename iT, typename uiT, typename vT>
void spmv_csr5_calibrate_kernel(const uiT* d_partition_pointer, vT* d_calibrator, vT* d_y, const iT p) {
    int num_thread = omp_get_max_threads();
    auto chunk = static_cast<int>(std::ceil((p - 1.) / num_thread));
    int stride_vT = ANONYMOUSLIB_X86_CACHELINE / sizeof(vT);
    // calculate the number of maximal active threads (for a static loop scheduling with size chunk)
    auto num_thread_active = static_cast<int>(std::ceil((p - 1.) / chunk));
    int num_cali = num_thread_active < num_thread ? num_thread_active : num_thread;

    for (int i = 0; i < num_cali; i++) {
        d_y[(d_partition_pointer[i * chunk] << 1) >> 1] += d_calibrator[i * stride_vT];
    }
}

template<typename iT, typename uiT, typename vT>
void spmv_csr5_tail_partition_kernel(const iT* d_row_pointer, const iT* d_column_index, const vT* d_value,
                                     const vT* d_x, vT* d_y, const iT tail_partition_start, const iT p, const iT m,
                                     const int sigma) {
    const iT index_first_element_tail = (p - 1) * ANONYMOUSLIB_CSR5_OMEGA * sigma;

#pragma omp parallel for
    for (iT row_id = tail_partition_start; row_id < m; row_id++) {
        const iT idx_start
          = row_id == tail_partition_start ? (p - 1) * ANONYMOUSLIB_CSR5_OMEGA * sigma : d_row_pointer[row_id];
        const iT idx_stop = d_row_pointer[row_id + 1];

        vT sum = 0;
        for (iT idx = idx_start; idx < idx_stop; idx++)
            sum += d_value[idx] * d_x[d_column_index[idx]]; // * alpha;

        if (row_id == tail_partition_start && d_row_pointer[row_id] != index_first_element_tail) {
            d_y[row_id] = d_y[row_id] + sum;
        } else {
            d_y[row_id] = sum;
        }
    }
}

template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT, typename ANONYMOUSLIB_VT>
int csr5_spmv(const int sigma, const ANONYMOUSLIB_IT p, const ANONYMOUSLIB_IT m, const int bit_y_offset,
              const int bit_scansum_offset, const int num_packet, const ANONYMOUSLIB_IT* row_pointer,
              const ANONYMOUSLIB_IT* column_index, const ANONYMOUSLIB_VT* value,
              const ANONYMOUSLIB_UIT* partition_pointer, const ANONYMOUSLIB_UIT* partition_descriptor,
              const ANONYMOUSLIB_IT* partition_descriptor_offset_pointer,
              const ANONYMOUSLIB_IT* partition_descriptor_offset, ANONYMOUSLIB_VT* calibrator,
              const ANONYMOUSLIB_IT tail_partition_start, const ANONYMOUSLIB_VT* x, ANONYMOUSLIB_VT* y) {
    int err = ANONYMOUSLIB_SUCCESS;

    // TODO: Switch this back to fill_n.
    memset(calibrator, 0, ANONYMOUSLIB_X86_CACHELINE * static_cast<size_t>(omp_get_max_threads()));
    spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>(
      column_index, value, x, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer,
      partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, sigma);

    spmv_csr5_calibrate_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>(partition_pointer, calibrator, y, p);

    spmv_csr5_tail_partition_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>(
      row_pointer, column_index, value, x, y, tail_partition_start, p, m, sigma);

    return err;
}

} // namespace csr5::avx2
#endif /* THIRD_PARTY_CSR5_INCLUDE_DETAIL_AVX2_CSR5_SPMV_AVX2 */
