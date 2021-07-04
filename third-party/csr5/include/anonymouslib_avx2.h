#ifndef THIRD_PARTY_CSR5_INCLUDE_ANONYMOUSLIB_AVX2
#define THIRD_PARTY_CSR5_INCLUDE_ANONYMOUSLIB_AVX2

#include "detail/avx2/utils_avx2.h"
#include "detail/common.h"
#include "detail/utils.h"

#include "detail/avx2/common_avx2.h"
#include "detail/avx2/csr5_spmv_avx2.h"
#include "detail/avx2/format_avx2.h"

#include <cstddef>
#include <utility>
#include <vector>

#include <dim/memory/aligned_allocator.h>
#include <fmt/chrono.h>
#include <fmt/format.h>

namespace csr5::avx2 {

template<typename Ty>
using cache_aligned_vector = std::vector<Ty, dim::memory::cache_aligned_allocator_t<Ty>>;

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
class anonymouslibHandle
{
    enum class format
    {
        csr,
        csr5
    };

public:
    anonymouslibHandle(size_t num_rows, size_t num_cols) : _num_rows{num_rows}, _num_cols{num_cols} {}
    int warmup();
    int inputCSR(size_t num_non_zero, ANONYMOUSLIB_IT* csr_row_pointer, ANONYMOUSLIB_IT* csr_column_index,
                 ANONYMOUSLIB_VT* csr_value);
    int asCSR() noexcept;
    int asCSR5() noexcept;
    int setX(ANONYMOUSLIB_VT* x);
    int spmv(ANONYMOUSLIB_VT alpha, ANONYMOUSLIB_VT* y);
    int destroy();
    void setSigma(size_t sigma);

private:
    size_t computeSigma();
    format _format{format::csr};
    size_t _num_rows{};
    size_t _num_cols{};
    size_t _num_non_zero{};

    ANONYMOUSLIB_IT* _csr_row_pointer;
    ANONYMOUSLIB_IT* _csr_column_index;
    ANONYMOUSLIB_VT* _csr_value;

    // TODO: do these need to be signed?
    size_t _csr5_sigma{};
    int _bit_y_offset{};
    int _bit_scansum_offset{};
    int _num_packet{};
    ANONYMOUSLIB_IT _tail_partition_start;

    ANONYMOUSLIB_IT _p;
    cache_aligned_vector<ANONYMOUSLIB_UIT> _csr5_partition_pointer;
    cache_aligned_vector<ANONYMOUSLIB_UIT> _csr5_partition_descriptor;

    ANONYMOUSLIB_IT _num_offsets;
    cache_aligned_vector<ANONYMOUSLIB_IT> _csr5_partition_descriptor_offset_pointer;
    cache_aligned_vector<ANONYMOUSLIB_IT> _csr5_partition_descriptor_offset;
    cache_aligned_vector<ANONYMOUSLIB_VT> _temp_calibrator;

    ANONYMOUSLIB_VT* _x;
};

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::warmup() {
    return ANONYMOUSLIB_SUCCESS;
}

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::inputCSR(size_t num_non_zero,
                                                                                     ANONYMOUSLIB_IT* csr_row_pointer,
                                                                                     ANONYMOUSLIB_IT* csr_column_index,
                                                                                     ANONYMOUSLIB_VT* csr_value) {
    _format = format::csr;
    _num_non_zero = num_non_zero;
    _csr_row_pointer = csr_row_pointer;
    _csr_column_index = csr_column_index;
    _csr_value = csr_value;

    return ANONYMOUSLIB_SUCCESS;
}

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::asCSR() noexcept {
    if (_format == format::csr)
        return ANONYMOUSLIB_SUCCESS;

    // convert csr5 data to csr data
    if (auto err = aosoa_transpose<false>(_csr5_sigma, _num_non_zero, _csr5_partition_pointer.data(), _csr_column_index,
                                          _csr_value);
        err != ANONYMOUSLIB_SUCCESS)
        return err;

    // free the two newly added CSR5 arrays
    std::exchange(_csr5_partition_pointer, {});
    std::exchange(_csr5_partition_descriptor, {});
    std::exchange(_temp_calibrator, {});
    std::exchange(_csr5_partition_descriptor_offset_pointer, {});

    if (_num_offsets)
        std::exchange(_csr5_partition_descriptor_offset, {});

    _format = format::csr;

    return ANONYMOUSLIB_SUCCESS;
}

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::asCSR5() noexcept {
    if (_format == format::csr5)
        return ANONYMOUSLIB_SUCCESS;

    // compute sigma
    _csr5_sigma = computeSigma();

    _bit_y_offset = static_cast<int>(get_needed_storage(ANONYMOUSLIB_CSR5_OMEGA * _csr5_sigma));
    _bit_scansum_offset = static_cast<int>(get_needed_storage(ANONYMOUSLIB_CSR5_OMEGA));

    if (_bit_y_offset + _bit_scansum_offset
        > static_cast<int>(sizeof(ANONYMOUSLIB_UIT) * 8 - 1)) // the 1st bit of bit-flag should be in the first packet
        return ANONYMOUSLIB_UNSUPPORTED_CSR5_OMEGA;

    // TODO: this shouldn't need the int cast when signedness is correct everywhere.
    _num_packet = static_cast<int>(
      std::ceil(static_cast<double>(_bit_y_offset + _bit_scansum_offset + static_cast<int>(_csr5_sigma))
                / static_cast<double>(sizeof(ANONYMOUSLIB_UIT) * 8)));

    // calculate the number of partitions
    _p = static_cast<ANONYMOUSLIB_IT>(
      std::ceil(static_cast<double>(_num_non_zero) / static_cast<double>(ANONYMOUSLIB_CSR5_OMEGA * _csr5_sigma)));

    // malloc the newly added arrays for CSR5
    _csr5_partition_pointer.resize(static_cast<size_t>(_p + 1));
    _csr5_partition_descriptor.resize(static_cast<size_t>(_p * ANONYMOUSLIB_CSR5_OMEGA * _num_packet));

    _temp_calibrator.resize(static_cast<size_t>(omp_get_max_threads())
                            * dim::memory::hardware_destructive_interference_size / sizeof(ANONYMOUSLIB_VT));

    _csr5_partition_descriptor_offset_pointer.resize(_csr5_partition_pointer.size());

    if (generate_partition_pointer<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>(
          static_cast<size_t>(_csr5_sigma), _num_non_zero, _csr5_partition_pointer,
          std::span{_csr_row_pointer, static_cast<size_t>(_num_rows + 1)})
        != ANONYMOUSLIB_SUCCESS)
        return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;

    _tail_partition_start = (_csr5_partition_pointer[static_cast<size_t>(_p - 1)] << 1) >> 1;

    // step 2. generate partition descriptor
    if (generate_partition_descriptor<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>(
          _csr5_sigma, _p, _bit_y_offset, _bit_scansum_offset, _num_packet, _csr_row_pointer,
          _csr5_partition_pointer.data(), _csr5_partition_descriptor.data(),
          _csr5_partition_descriptor_offset_pointer.data(), &_num_offsets)
        != ANONYMOUSLIB_SUCCESS)
        return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;

    if (_num_offsets) {
        _csr5_partition_descriptor_offset.resize(static_cast<size_t>(_num_offsets));
        if (generate_partition_descriptor_offset<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>(
              _csr5_sigma, _p, _bit_y_offset, _bit_scansum_offset, _num_packet, _csr_row_pointer,
              _csr5_partition_pointer.data(), _csr5_partition_descriptor.data(),
              _csr5_partition_descriptor_offset_pointer.data(), _csr5_partition_descriptor_offset.data())
            != ANONYMOUSLIB_SUCCESS)
            return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
    }

    if (aosoa_transpose<true>(_csr5_sigma, _num_non_zero, _csr5_partition_pointer.data(), _csr_column_index, _csr_value)
        != ANONYMOUSLIB_SUCCESS)
        return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;

    _format = format::csr5;

    return ANONYMOUSLIB_SUCCESS;
}

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::setX(ANONYMOUSLIB_VT* x) {
    int err = ANONYMOUSLIB_SUCCESS;

    _x = x;

    return err;
}

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::spmv(const ANONYMOUSLIB_VT /*alpha*/,
                                                                                 ANONYMOUSLIB_VT* y) {
    if (_format == format::csr)
        return ANONYMOUSLIB_UNSUPPORTED_CSR_SPMV;

    csr5_spmv<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>(
      _csr5_sigma, _p, _num_rows, _bit_y_offset, _bit_scansum_offset, _num_packet, _csr_row_pointer, _csr_column_index,
      _csr_value, _csr5_partition_pointer.data(), _csr5_partition_descriptor.data(),
      _csr5_partition_descriptor_offset_pointer.data(), _csr5_partition_descriptor_offset.data(),
      _temp_calibrator.data(), _tail_partition_start, _x, y);

    return ANONYMOUSLIB_SUCCESS;
}

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::destroy() {
    return asCSR();
}

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
void anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::setSigma(size_t sigma) {
    _csr5_sigma = sigma;
}

template<class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
size_t anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::computeSigma() {
    return _csr5_sigma;
}

} // namespace csr5::avx2

#endif /* THIRD_PARTY_CSR5_INCLUDE_ANONYMOUSLIB_AVX2 */
