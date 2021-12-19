#ifndef INCLUDE_DIM_BIT
#define INCLUDE_DIM_BIT

#include <climits>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace dim {

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
    return Integral{1} << (bit_size<Integral> - 1 - which);
}

/**
 * @brief Set the bit starting at the least significant bit.
 *
 * @tparam Integral any integral type.
 * @param which the bit to be set (0 means LSB, 1 means LSB + 1 etc.)
 * @return constexpr Integral a value with needed bit set.
 */
template<typename Integral>
constexpr Integral set_rbit(size_t which) {
    return Integral{1} << which;
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

/**
 * @brief Checks if a bit is set.
 *
 * @tparam Integral any integral type.
 * @param which the bit to be set (0 means LSB, 1 means LSB + 1 etc.)
 * @return constexpr bool true if bit is set, false otherwise.
 */
template<typename Integral>
constexpr bool has_rbit_set(Integral value, size_t which) {
    return value & set_rbit<Integral>(which);
}

template<typename Integral>
constexpr Integral msb = set_bit<Integral>(0);

template<std::integral Integral>
constexpr Integral all_bits_set = ~Integral{0};

} // namespace dim

#endif /* INCLUDE_DIM_BIT */
