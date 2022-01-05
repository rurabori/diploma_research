#ifndef INCLUDE_DIM_MAT_STORAGE_FORMATS_CSR5_CALIBRATOR
#define INCLUDE_DIM_MAT_STORAGE_FORMATS_CSR5_CALIBRATOR

#include <dim/memory/aligned_allocator.h>
#include <omp.h>

namespace dim::mat {

/**
 * @brief Calibrator class to hold values of non-owning elements.
 *
 * @tparam ValueType same as the CSR5 matrix that will use it.
 */
template<typename ValueType>
class calibrator_t
{
    //! @brief stores one calibrator element per cache-line.
    struct element_t
    {
        alignas(memory::hardware_destructive_interference_size) ValueType value{};
    };

    std::vector<element_t> _elements;

public:
    explicit calibrator_t(size_t num_elements) : _elements(num_elements) {}
    calibrator_t() : calibrator_t{static_cast<size_t>(::omp_get_max_threads())} {}

    auto operator[](size_t index) noexcept -> ValueType& { return _elements[index].value; }
    auto operator[](size_t index) const noexcept -> ValueType { return _elements[index].value; }
};

} // namespace dim::mat

#endif /* INCLUDE_DIM_MAT_STORAGE_FORMATS_CSR5_CALIBRATOR */
