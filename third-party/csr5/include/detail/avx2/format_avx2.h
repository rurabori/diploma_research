#ifndef THIRD_PARTY_CSR5_INCLUDE_DETAIL_AVX2_FORMAT_AVX2
#define THIRD_PARTY_CSR5_INCLUDE_DETAIL_AVX2_FORMAT_AVX2

#include "anonymouslib_avx2.h"
#include "common_avx2.h"
#include "utils_avx2.h"

#include <algorithm>
#include <cstddef>
#include <dim/memory/aligned_allocator.h>
#include <span>

namespace csr5::avx2 {

template<typename iT, typename uiT>
void generate_partition_pointer_s1_kernel(std::span<const iT> row_pointer, const std::span<uiT> partition,
                                          const size_t sigma, const size_t nnz) {
#pragma omp parallel for
    for (size_t global_id = 0; global_id < partition.size(); global_id++) {
        // compute partition boundaries by partition of size sigma * omega
        auto boundary = static_cast<iT>(global_id * sigma * ANONYMOUSLIB_CSR5_OMEGA);

        // clamp partition boundaries to [0, nnz]
        boundary = boundary > nnz ? nnz : boundary;

        // binary search
        partition[global_id]
          = static_cast<uiT>(
              std::distance(row_pointer.begin(), std::upper_bound(row_pointer.begin(), row_pointer.end(), boundary)))
            - 1;
    }
}

template<typename iT>
bool is_dirty(const std::span<const iT> row) {
    for (size_t idx = 1; idx < row.size(); ++idx)
        if (row[idx - 1] == row[idx])
            return true;

    return false;
}

template<typename iT, typename uiT>
void generate_partition_pointer_s2_kernel(const std::span<const iT> row, const std::span<uiT> partition) {
#pragma omp parallel for
    for (size_t group_id = 1; group_id < partition.size(); group_id++) {
        auto start = (partition[group_id - 1] << 1) >> 1;
        auto stop = (partition[group_id] << 1) >> 1;

        if (start == stop)
            continue;

        if (is_dirty(row.subspan(start, stop - start))) {
            start |= uiT{1} << (sizeof(uiT) * 8 - 1); // TODO: investigate why the MSB is set.
            partition[group_id - 1] = start;
        }
    }
}

template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT>
int generate_partition_pointer(const size_t sigma, const size_t num_non_zero,
                               const std::span<ANONYMOUSLIB_UIT> partition,
                               const std::span<const ANONYMOUSLIB_IT> row_pointer) {
    // step 1. binary search row pointer
    generate_partition_pointer_s1_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>(row_pointer, partition, sigma,
                                                                            num_non_zero);

    // step 2. check empty rows
    generate_partition_pointer_s2_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>(row_pointer, partition);

    printf("%d\n", row_pointer.size());

    return ANONYMOUSLIB_SUCCESS;
}

template<typename iT, typename uiT>
void generate_partition_descriptor_s1_kernel(const iT* d_row_pointer, const uiT* d_partition_pointer,
                                             uiT* d_partition_descriptor, const iT p, const size_t sigma,
                                             const int bit_all_offset, const int num_packet) {
#pragma omp parallel for
    for (int par_id = 0; par_id < p - 1; par_id++) {
        const iT row_start = d_partition_pointer[par_id] & 0x7FFFFFFF;
        const iT row_stop = d_partition_pointer[par_id + 1] & 0x7FFFFFFF;

        for (int rid = row_start; rid <= row_stop; rid++) {
            int ptr = d_row_pointer[rid];
            int pid = ptr / (ANONYMOUSLIB_CSR5_OMEGA * sigma);

            if (pid == par_id) {
                int lx = (ptr / sigma) % ANONYMOUSLIB_CSR5_OMEGA;

                const int glid = ptr % sigma + bit_all_offset;
                const int ly = glid / 32;
                const int llid = glid % 32;

                const uiT val = 0x1 << (31 - llid);

                const int location = pid * ANONYMOUSLIB_CSR5_OMEGA * num_packet + ly * ANONYMOUSLIB_CSR5_OMEGA + lx;
                d_partition_descriptor[location] |= val;
            }
        }
    }
}

template<typename iT, typename uiT>
void generate_partition_descriptor_s2_kernel(const uiT* d_partition_pointer, uiT* d_partition_descriptor,
                                             iT* d_partition_descriptor_offset_pointer, const int sigma,
                                             const int num_packet, const int bit_y_offset, const int bit_scansum_offset,
                                             const iT p) {
    const auto num_thread = static_cast<size_t>(omp_get_max_threads());

    dim::memory::cache_aligned_vector<int> s_segn_scan_all(2 * ANONYMOUSLIB_CSR5_OMEGA * num_thread);
    dim::memory::cache_aligned_vector<int> s_present_all(2 * ANONYMOUSLIB_CSR5_OMEGA * num_thread);
    for (size_t i = 0; i < num_thread; i++)
        s_present_all[i * 2 * ANONYMOUSLIB_CSR5_OMEGA + ANONYMOUSLIB_CSR5_OMEGA] = 1;

    const int bit_all_offset = bit_y_offset + bit_scansum_offset;

#pragma omp parallel for
    for (int par_id = 0; par_id < p - 1; par_id++) {
        const auto segment_start = static_cast<size_t>(omp_get_thread_num()) * 2 * ANONYMOUSLIB_CSR5_OMEGA;
        int* s_segn_scan = &s_segn_scan_all[segment_start];
        int* s_present = &s_present_all[segment_start];

        memset(s_segn_scan, 0, (ANONYMOUSLIB_CSR5_OMEGA + 1) * sizeof(int));
        memset(s_present, 0, ANONYMOUSLIB_CSR5_OMEGA * sizeof(int));

        bool with_empty_rows = (d_partition_pointer[par_id] >> 31) & 0x1;
        iT row_start = d_partition_pointer[par_id] & 0x7FFFFFFF;
        const iT row_stop = d_partition_pointer[par_id + 1] & 0x7FFFFFFF;

        if (row_start == row_stop)
            continue;

#pragma omp simd
        for (int lane_id = 0; lane_id < ANONYMOUSLIB_CSR5_OMEGA; lane_id++) {
            int start = 0;
            int stop = 0;
            int segn = 0;
            bool present = 0;
            uiT bitflag = 0;

            present |= !lane_id;

            // extract the first bit-flag packet
            int ly = 0;
            uiT first_packet = d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet + lane_id];
            bitflag = (first_packet << bit_all_offset) | (static_cast<uiT>(present) << 31);
            start = !((bitflag >> 31) & 0x1);
            present |= (bitflag >> 31) & 0x1;

            for (int i = 1; i < sigma; i++) {
                if ((!ly && i == 32 - bit_all_offset) || (ly && (i - (32 - bit_all_offset)) % 32 == 0)) {
                    ly++;
                    bitflag = d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet
                                                     + ly * ANONYMOUSLIB_CSR5_OMEGA + lane_id];
                }
                const int norm_i = !ly ? i : i - (32 - bit_all_offset);
                stop += (bitflag >> (31 - norm_i % 32)) & 0x1;
                present |= (bitflag >> (31 - norm_i % 32)) & 0x1;
            }

            // compute y_offset for all partitions
            segn = stop - start + present;
            segn = segn > 0 ? segn : 0;

            s_segn_scan[lane_id] = segn;

            // compute scansum_offset
            s_present[lane_id] = present;
        }

        scan_single<int>(s_segn_scan, ANONYMOUSLIB_CSR5_OMEGA + 1);

        if (with_empty_rows) {
            d_partition_descriptor_offset_pointer[par_id] = s_segn_scan[ANONYMOUSLIB_CSR5_OMEGA];
            d_partition_descriptor_offset_pointer[p] += s_segn_scan[ANONYMOUSLIB_CSR5_OMEGA];
        }

#pragma omp simd
        for (int lane_id = 0; lane_id < ANONYMOUSLIB_CSR5_OMEGA; lane_id++) {
            // TODO: check the signedness requirements here.
            auto y_offset = static_cast<uiT>(s_segn_scan[lane_id]);

            uiT scansum_offset = 0;
            int next1 = lane_id + 1;
            if (s_present[lane_id]) {
                while (!s_present[next1] && next1 < ANONYMOUSLIB_CSR5_OMEGA) {
                    scansum_offset++;
                    next1++;
                }
            }

            uiT first_packet = d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet + lane_id];

            y_offset = lane_id ? y_offset - 1 : 0;

            first_packet |= y_offset << (32 - bit_y_offset);
            first_packet |= scansum_offset << (32 - bit_all_offset);

            d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet + lane_id] = first_packet;
        }
    }
}

template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT>
int generate_partition_descriptor(const size_t sigma, const ANONYMOUSLIB_IT p, const int bit_y_offset,
                                  const int bit_scansum_offset, const int num_packet,
                                  const ANONYMOUSLIB_IT* row_pointer, const ANONYMOUSLIB_UIT* partition_pointer,
                                  ANONYMOUSLIB_UIT* partition_descriptor,
                                  ANONYMOUSLIB_IT* partition_descriptor_offset_pointer, ANONYMOUSLIB_IT* _num_offsets) {
    int bit_all_offset = bit_y_offset + bit_scansum_offset;

    generate_partition_descriptor_s1_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>(
      row_pointer, partition_pointer, partition_descriptor, p, sigma, bit_all_offset, num_packet);

    generate_partition_descriptor_s2_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>(
      partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, sigma, num_packet, bit_y_offset,
      bit_scansum_offset, p);

    if (partition_descriptor_offset_pointer[p])
        scan_single<ANONYMOUSLIB_IT>(partition_descriptor_offset_pointer, p + 1);

    *_num_offsets = partition_descriptor_offset_pointer[p];

    // print for debug
    //    cout << "partition_descriptor(1) = " << endl;
    //    print_tile<ANONYMOUSLIB_UIT>(partition_descriptor, num_packet, ANONYMOUSLIB_CSR5_OMEGA);
    //    cout << "partition_descriptor(2) = " << endl;
    //    print_tile<ANONYMOUSLIB_UIT>(&partition_descriptor[num_packet * ANONYMOUSLIB_CSR5_OMEGA], num_packet,
    //    ANONYMOUSLIB_CSR5_OMEGA);

    return ANONYMOUSLIB_SUCCESS;
}

template<typename iT, typename uiT>
void generate_partition_descriptor_offset_kernel(const iT* d_row_pointer, const uiT* d_partition_pointer,
                                                 const uiT* d_partition_descriptor,
                                                 const iT* d_partition_descriptor_offset_pointer,
                                                 iT* d_partition_descriptor_offset, const iT p, const int num_packet,
                                                 const int bit_y_offset, const int bit_scansum_offset,
                                                 const int c_sigma) {
    const int bit_all_offset = bit_y_offset + bit_scansum_offset;
    const int bit_bitflag = 32 - bit_all_offset;

#pragma omp parallel for
    for (int par_id = 0; par_id < p - 1; par_id++) {
        bool with_empty_rows = (d_partition_pointer[par_id] >> 31) & 0x1;
        if (!with_empty_rows)
            continue;

        const auto row_start = d_partition_pointer[par_id] & 0x7FFFFFFF;
        const auto row_stop = d_partition_pointer[par_id + 1] & 0x7FFFFFFF;

        // TODO: check if offsets could be negative (C implicit conversion makes this unsigned in operations anyway).
        const auto offset_pointer = static_cast<size_t>(d_partition_descriptor_offset_pointer[par_id]);

#pragma omp simd
        for (int lane_id = 0; lane_id < ANONYMOUSLIB_CSR5_OMEGA; lane_id++) {
            bool local_bit{};

            // extract the first bit-flag packet
            int ly = 0;
            uiT descriptor = d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet + lane_id];
            uiT y_offset = descriptor >> (32 - bit_y_offset);

            descriptor = descriptor << bit_all_offset;
            descriptor = lane_id ? descriptor : descriptor | 0x80000000;

            local_bit = (descriptor >> 31) & 0x1;

            if (local_bit && lane_id) {
                const iT idx = par_id * ANONYMOUSLIB_CSR5_OMEGA * c_sigma + lane_id * c_sigma;
                const auto y_index = static_cast<iT>(binary_search_right_boundary_kernel<iT>(
                                       &d_row_pointer[row_start + 1], idx, row_stop - row_start))
                                     - 1;

                d_partition_descriptor_offset[offset_pointer + y_offset] = y_index;

                y_offset++;
            }

            for (int i = 1; i < c_sigma; i++) {
                if ((!ly && i == bit_bitflag) || (ly && !(31 & (i - bit_bitflag)))) {
                    ly++;
                    descriptor = d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet
                                                        + ly * ANONYMOUSLIB_CSR5_OMEGA + lane_id];
                }
                const int norm_i = 31 & (!ly ? i : i - bit_bitflag);

                local_bit = (descriptor >> (31 - norm_i)) & 0x1;

                if (local_bit) {
                    const iT idx = par_id * ANONYMOUSLIB_CSR5_OMEGA * c_sigma + lane_id * c_sigma + i;
                    const auto y_index = static_cast<int>(binary_search_right_boundary_kernel<iT>(
                                           &d_row_pointer[row_start + 1], idx, row_stop - row_start))
                                         - 1;

                    d_partition_descriptor_offset[offset_pointer + y_offset] = y_index;

                    y_offset++;
                }
            }
        }
    }
}

template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT>
int generate_partition_descriptor_offset(const int sigma, const ANONYMOUSLIB_IT p, const int bit_y_offset,
                                         const int bit_scansum_offset, const int num_packet,
                                         const ANONYMOUSLIB_IT* row_pointer, const ANONYMOUSLIB_UIT* partition_pointer,
                                         ANONYMOUSLIB_UIT* partition_descriptor,
                                         ANONYMOUSLIB_IT* partition_descriptor_offset_pointer,
                                         ANONYMOUSLIB_IT* partition_descriptor_offset) {
    generate_partition_descriptor_offset_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>(
      row_pointer, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer,
      partition_descriptor_offset, p, num_packet, bit_y_offset, bit_scansum_offset, sigma);

    return ANONYMOUSLIB_SUCCESS;
}

template<bool R2C, typename T, typename uiT> // R2C==true means CSR->CSR5, otherwise CSR5->CSR
void aosoa_transpose_kernel_smem(T* d_data, const uiT* d_partition_pointer, const size_t nnz, const size_t sigma) {
    const auto transform_index = [&](size_t idx, bool r2c) {
        if (r2c)
            return std::pair{(idx / sigma), (idx % sigma)};

        return std::pair{(idx % ANONYMOUSLIB_CSR5_OMEGA), (idx / ANONYMOUSLIB_CSR5_OMEGA)};
    };

    const auto num_p
      = static_cast<size_t>(std::ceil(static_cast<double>(nnz) / static_cast<double>(ANONYMOUSLIB_CSR5_OMEGA * sigma)))
        - 1;
    const size_t size_base = sigma * ANONYMOUSLIB_CSR5_OMEGA;

    dim::memory::cache_aligned_vector<T> s_data_all(size_base * static_cast<size_t>(omp_get_max_threads()));

#pragma omp parallel for
    for (size_t par_id = 0; par_id < num_p; ++par_id) {
        T* s_data = &s_data_all[size_base * static_cast<size_t>(omp_get_thread_num())];

        // if this is fast track partition, do not transpose it
        if (d_partition_pointer[par_id] == d_partition_pointer[par_id + 1])
            continue;

            // load global data to shared mem

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
    }
}

template<bool R2C, typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT, typename ANONYMOUSLIB_VT>
int aosoa_transpose(const size_t sigma, const size_t nnz, const ANONYMOUSLIB_UIT* partition_pointer,
                    ANONYMOUSLIB_IT* column_index, ANONYMOUSLIB_VT* value) {
    aosoa_transpose_kernel_smem<R2C, ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>(column_index, partition_pointer, nnz, sigma);
    aosoa_transpose_kernel_smem<R2C, ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT>(value, partition_pointer, nnz, sigma);

    return ANONYMOUSLIB_SUCCESS;
}

} // namespace csr5::avx2

#endif /* THIRD_PARTY_CSR5_INCLUDE_DETAIL_AVX2_FORMAT_AVX2 */
