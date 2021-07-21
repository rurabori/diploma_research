#ifndef THIRD_PARTY_CSR5_INCLUDE_DETAIL_AVX2_CSR5_SPMV_AVX2
#define THIRD_PARTY_CSR5_INCLUDE_DETAIL_AVX2_CSR5_SPMV_AVX2

#include "common_avx2.h"
#include "detail/avx2/format_avx2.h"
#include "utils_avx2.h"

#include <bit>

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

template<typename iT, typename uiT, typename vT>
void spmv_csr5_compute_kernel(const iT* d_column_index, const vT* d_value, const vT* d_x,
                              const uiT* d_partition_pointer, const uiT* d_partition_descriptor,
                              const iT* d_partition_descriptor_offset_pointer, const iT* d_partition_descriptor_offset,
                              vT* d_calibrator, vT* d_y, const iT p, const int num_packet, const int bit_y_offset,
                              const int bit_scansum_offset, const int c_sigma) {
    const int num_thread = omp_get_max_threads();
    const auto chunk = static_cast<int>(std::ceil(static_cast<double>(p - 1) / static_cast<double>(num_thread)));
    const int stride_vT = ANONYMOUSLIB_X86_CACHELINE / sizeof(vT);
    const auto num_thread_active = static_cast<int>(std::ceil((p - 1.0) / chunk));

#pragma omp parallel
    {
        const auto tid = omp_get_thread_num();
        iT start_row_start = tid < num_thread_active ? strip_dirty(d_partition_pointer[tid * chunk]) : 0;

        // these need to be aligned to 32 as the intrinsics require it.
        alignas(32) vT s_sum[8];        // allocate a cache line
        alignas(32) vT s_first_sum[8];  // allocate a cache line
        alignas(32) uint64_t s_cond[8]; // allocate a cache line
        alignas(32) int s_y_idx[16];    // allocate a cache line

        __m256d sum256d = _mm256_setzero_pd();
        __m256d tmp_sum256d = _mm256_setzero_pd();
        __m256d first_sum256d = _mm256_setzero_pd();
        __m256d last_sum256d = _mm256_setzero_pd();
        __m128i scansum_offset128i, y_offset128i, y_idx128i;
        __m256i start256i;
        __m256i stop256i = _mm256_setzero_si256();

        __m256d value256d, x256d;

        __m256i local_bit256i;
        __m256i direct256i;

        __m128i descriptor128i;
        __m256i tmp256i;

#pragma omp for schedule(static, chunk)
        for (int par_id = 0; par_id < p - 1; par_id++) {
            const auto partition_offset_base = par_id * ANONYMOUSLIB_CSR5_OMEGA;
            const auto partition_descriptor_offset = partition_offset_base * num_packet;

            const iT* d_column_index_partition = &d_column_index[partition_offset_base * c_sigma];
            const vT* d_value_partition = &d_value[partition_offset_base * c_sigma];

            // TODO: check if the signedness used is needed.
            auto row_start = static_cast<iT>(d_partition_pointer[par_id]);
            const iT row_stop = d_partition_pointer[par_id + 1] & 0x7FFFFFFF;

            if (row_start == row_stop) // fast track through reduction
            {
                // check whether the the partition contains the first element of row "row_start"
                // => we are the first writing data to d_y[row_start]
                bool fast_direct
                  = has_bit_set(d_partition_descriptor[partition_descriptor_offset], bit_y_offset + bit_scansum_offset);
                partition_fast_track<iT, vT>(d_value_partition, d_x, d_column_index_partition, d_calibrator, d_y,
                                             row_start, tid, start_row_start, stride_vT, fast_direct);
            } else // normal track for all the other partitions
            {
                const bool empty_rows = (row_start >> 31) & 0x1;
                row_start &= 0x7FFFFFFF;

                vT* d_y_local = &d_y[row_start + 1];
                const int offset_pointer = empty_rows ? d_partition_descriptor_offset_pointer[par_id] : 0;

                // TODO: do we really need to cast away const here?
                auto d_partition_descriptor128i = const_cast<__m128i*>(
                  reinterpret_cast<const __m128i*>(&d_partition_descriptor[partition_descriptor_offset]));

                first_sum256d = _mm256_setzero_pd();
                stop256i = _mm256_setzero_si256();

                descriptor128i = _mm_load_si128(d_partition_descriptor128i);

                y_offset128i = _mm_srli_epi32(descriptor128i, 32 - bit_y_offset);
                scansum_offset128i = _mm_slli_epi32(descriptor128i, bit_y_offset);
                scansum_offset128i = _mm_srli_epi32(scansum_offset128i, 32 - bit_scansum_offset);

                descriptor128i = _mm_slli_epi32(descriptor128i, bit_y_offset + bit_scansum_offset);

                // remember if the first element of this partition is the first element of a new row
                local_bit256i = _mm256_cvtepu32_epi64(_mm_srli_epi32(descriptor128i, 31));
                bool first_direct = false;
                _mm256_store_si256(reinterpret_cast<__m256i*>(std::data(s_cond)), local_bit256i);
                if (s_cond[0])
                    first_direct = true;

                // remember if the first element of the first partition of the current thread is the first element of a
                // new row
                bool first_all_direct = false;
                if (par_id == tid * chunk)
                    first_all_direct = first_direct;

                descriptor128i = _mm_or_si128(descriptor128i, _mm_set_epi32(0, 0, 0, std::bit_cast<int>(0x80000000)));

                local_bit256i = _mm256_cvtepu32_epi64(_mm_srli_epi32(descriptor128i, 31));
                start256i = _mm256_sub_epi64(_mm256_set1_epi64x(0x1), local_bit256i);
                direct256i = _mm256_and_si256(local_bit256i, _mm256_set_epi64x(0x1, 0x1, 0x1, 0));

                value256d = _mm256_load_pd(d_value_partition);

                x256d = load_x(d_x, d_column_index_partition, 0);

                sum256d = _mm256_mul_pd(value256d, x256d);

                // step 1. thread-level seg sum
#if ANONYMOUSLIB_CSR5_SIGMA > 23
                int ly = 0;
#endif

                for (int i = 1; i < ANONYMOUSLIB_CSR5_SIGMA; i++) {
                    x256d = load_x(d_x, d_column_index_partition, i * ANONYMOUSLIB_CSR5_OMEGA);

#if ANONYMOUSLIB_CSR5_SIGMA > 23
                    int norm_i = i - (32 - bit_y_offset - bit_scansum_offset);

                    if (!(ly || norm_i) || (ly && !(norm_i % 32))) {
                        ly++;
                        descriptor128i = _mm_load_si128(&d_partition_descriptor128i[ly]);
                    }
                    norm_i = !ly ? i : norm_i;
                    norm_i = 31 - norm_i % 32;

                    local_bit256i = _mm256_and_si256(_mm256_cvtepu32_epi64(_mm_srli_epi32(descriptor128i, norm_i)),
                                                     _mm256_set1_epi64x(0x1));
#else
                    local_bit256i = _mm256_and_si256(_mm256_cvtepu32_epi64(_mm_srli_epi32(descriptor128i, 31 - i)),
                                                     _mm256_set1_epi64x(0x1));
#endif

                    int store_to_offchip = _mm256_testz_si256(
                      local_bit256i, _mm256_set1_epi64x(std::bit_cast<int64_t>(0xFFFFFFFFFFFFFFFF)));

                    if (!store_to_offchip) {
                        y_idx128i = empty_rows ? _mm_i32gather_epi32(&d_partition_descriptor_offset[offset_pointer],
                                                                     y_offset128i, 4)
                                               : y_offset128i;

                        // mask scatter store
                        _mm_store_si128(reinterpret_cast<__m128i*>(std::data(s_y_idx)), y_idx128i);
                        _mm256_store_pd(s_sum, sum256d);
                        _mm256_store_si256(reinterpret_cast<__m256i*>(std::data(s_cond)),
                                           _mm256_and_si256(direct256i, local_bit256i));
                        int inc0 = 0;
                        int inc1 = 0;
                        int inc2 = 0;
                        int inc3 = 0;
                        if (s_cond[0]) {
                            d_y_local[s_y_idx[0]] = s_sum[0];
                            inc0 = 1;
                        }
                        if (s_cond[1]) {
                            d_y_local[s_y_idx[1]] = s_sum[1];
                            inc1 = 1;
                        }
                        if (s_cond[2]) {
                            d_y_local[s_y_idx[2]] = s_sum[2];
                            inc2 = 1;
                        }
                        if (s_cond[3]) {
                            d_y_local[s_y_idx[3]] = s_sum[3];
                            inc3 = 1;
                        }

                        y_offset128i = _mm_add_epi32(y_offset128i, _mm_set_epi32(inc3, inc2, inc1, inc0));

                        tmp256i = _mm256_andnot_si256(_mm256_cmpeq_epi64(direct256i, _mm256_set1_epi64x(0x1)),
                                                      _mm256_cmpeq_epi64(local_bit256i, _mm256_set1_epi64x(0x1)));
                        first_sum256d = _mm256_add_pd(
                          _mm256_and_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(tmp256i, _mm256_set1_epi64x(0))),
                                        first_sum256d),
                          _mm256_and_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(
                                          tmp256i, _mm256_set1_epi64x(std::bit_cast<int64_t>(0xFFFFFFFFFFFFFFFF)))),
                                        sum256d));

                        sum256d = _mm256_and_pd(
                          _mm256_castsi256_pd(_mm256_cmpeq_epi64(local_bit256i, _mm256_set1_epi64x(0))), sum256d);

                        direct256i = _mm256_or_si256(direct256i, local_bit256i);
                        stop256i = _mm256_add_epi64(stop256i, local_bit256i);
                    }

                    value256d = _mm256_load_pd(&d_value_partition[i * ANONYMOUSLIB_CSR5_OMEGA]);
                    sum256d = _mm256_fmadd_pd(value256d, x256d, sum256d);
                }

                tmp256i = _mm256_cmpeq_epi64(direct256i, _mm256_set1_epi64x(0x1));

                first_sum256d = _mm256_and_pd(_mm256_castsi256_pd(tmp256i), first_sum256d);
                tmp256i = _mm256_cmpeq_epi64(tmp256i, _mm256_set1_epi64x(0));
                first_sum256d = _mm256_add_pd(first_sum256d, _mm256_and_pd(_mm256_castsi256_pd(tmp256i), sum256d));

                last_sum256d = sum256d;

                tmp256i = _mm256_cmpeq_epi64(start256i, _mm256_set1_epi64x(0x1));
                sum256d = _mm256_and_pd(_mm256_castsi256_pd(tmp256i), first_sum256d);

                sum256d = _mm256_permute4x64_pd(sum256d, 0x39);
                sum256d = _mm256_and_pd(
                  _mm256_castsi256_pd(_mm256_setr_epi64x(
                    std::bit_cast<int64_t>(0xFFFFFFFFFFFFFFFF), std::bit_cast<int64_t>(0xFFFFFFFFFFFFFFFF),
                    std::bit_cast<int64_t>(0xFFFFFFFFFFFFFFFF), 0x0000000000000000)),
                  sum256d);

                tmp_sum256d = sum256d;
                sum256d = hscan_avx(sum256d);

                scansum_offset128i = _mm_add_epi32(scansum_offset128i, _mm_set_epi32(3, 2, 1, 0));

                tmp256i = _mm256_castsi128_si256(scansum_offset128i);
                tmp256i = _mm256_permutevar8x32_epi32(tmp256i, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
                tmp256i = _mm256_add_epi32(tmp256i, tmp256i);
                tmp256i = _mm256_add_epi32(tmp256i, _mm256_set_epi32(1, 0, 1, 0, 1, 0, 1, 0));

                sum256d = _mm256_sub_pd(
                  _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(sum256d), tmp256i)), sum256d);
                sum256d = _mm256_add_pd(sum256d, tmp_sum256d);

                tmp256i = _mm256_cmpgt_epi64(start256i, stop256i);
                tmp256i = _mm256_cmpeq_epi64(tmp256i, _mm256_set1_epi64x(0));

                last_sum256d = _mm256_add_pd(last_sum256d, _mm256_and_pd(_mm256_castsi256_pd(tmp256i), sum256d));

                y_idx128i = empty_rows
                              ? _mm_i32gather_epi32(&d_partition_descriptor_offset[offset_pointer], y_offset128i, 4)
                              : y_offset128i;

                _mm256_store_si256(reinterpret_cast<__m256i*>(std::data(s_cond)), direct256i);
                _mm_store_si128(reinterpret_cast<__m128i*>(std::data(s_y_idx)), y_idx128i);
                _mm256_store_pd(s_sum, last_sum256d);

                if (s_cond[0]) {
                    d_y_local[s_y_idx[0]] = s_sum[0];
                    _mm256_store_pd(s_first_sum, first_sum256d);
                }
                if (s_cond[1])
                    d_y_local[s_y_idx[1]] = s_sum[1];
                if (s_cond[2])
                    d_y_local[s_y_idx[2]] = s_sum[2];
                if (s_cond[3])
                    d_y_local[s_y_idx[3]] = s_sum[3];

                // only use calibrator if this partition does not contain the first element of the row "row_start"
                if (row_start == start_row_start && !first_all_direct)
                    d_calibrator[tid * stride_vT] += s_cond[0] ? s_first_sum[0] : s_sum[0];
                else {
                    if (first_direct)
                        d_y[row_start] = s_cond[0] ? s_first_sum[0] : s_sum[0];
                    else
                        d_y[row_start] += s_cond[0] ? s_first_sum[0] : s_sum[0];
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
