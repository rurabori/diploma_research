#ifndef INCLUDE_DIM_VEC
#define INCLUDE_DIM_VEC

#include <dim/memory/aligned_allocator.h>

#include <algorithm>
#include <execution>
#include <span>

namespace dim {

/**
 * @brief The base class for both vec and vec_view to abstract common operations.
 *
 * @tparam ValueType to be used.
 * @tparam StorageContainer for the values to be stored in.
 */
template<typename ValueType, template<typename> typename StorageContainer = memory::cache_aligned_vector>
class vec_impl
{
protected:
    using container_t = StorageContainer<ValueType>;

    /**
     * @brief The base for y = alpha * [] + [] operations.
     */
    auto a_p_impl(std::span<const ValueType> x, auto&& op) noexcept -> void {
        std::transform(std::execution::par_unseq, std::begin(_values), std::end(_values), std::begin(x),
                       std::begin(_values), std::forward<decltype(op)>(op));
    }

private:
    container_t _values{};

public:
    explicit vec_impl(container_t&& values) : _values{std::move(values)} {}

    /**
     * @brief Returns a view to the underlying data.
     */
    auto raw() noexcept -> std::span<ValueType> { return _values; }
    auto raw() const noexcept -> std::span<const ValueType> { return _values; }

    /**
     * @brief Performs y = alpha * x + y. With this object being the y.
     */
    template<template<typename> typename StorageContainerR>
    auto axpy(const vec_impl<ValueType, StorageContainerR>& x, ValueType alpha) noexcept -> void {
        a_p_impl(x.raw(), [alpha](ValueType lhs_elem, ValueType rhs_elem) { return lhs_elem + alpha * rhs_elem; });
    }

    /**
     * @brief Performs y = alpha * y + x. With this object being the y.
     */
    template<template<typename> typename StorageContainerR>
    auto aypx(const vec_impl<ValueType, StorageContainerR>& x, ValueType alpha) noexcept -> void {
        a_p_impl(x.raw(), [alpha](ValueType lhs_elem, ValueType rhs_elem) { return rhs_elem + alpha * lhs_elem; });
    }

    /**
     * @brief Sets the whole vector to [value].
     */
    auto set(ValueType value) noexcept -> void {
        std::fill(std::execution::par_unseq, _values.begin(), _values.end(), value);
    }

    /**
     * @brief Returns the size of the vector.
     */
    [[nodiscard]] auto size() const noexcept -> size_t { return _values.size(); }
};

/**
 * @brief Performs dot product of [l] and [r].
 */
template<typename ValueType, template<typename> typename StorageContainerL,
         template<typename> typename StorageContainerR>
auto dot(const vec_impl<ValueType, StorageContainerL>& l, const vec_impl<ValueType, StorageContainerR>& r)
  -> ValueType {
    return std::transform_reduce(std::execution::par_unseq, l.raw().begin(), l.raw().end(), r.raw().begin(),
                                 ValueType{});
}

/**
 * @brief A view into an owning vec.
 */
template<typename ValueType>
class vec_view : public vec_impl<ValueType, std::span>
{
    using parent_t = vec_impl<ValueType, std::span>;

public:
    explicit vec_view(std::span<ValueType> vals) : parent_t(std::move(vals)) {}

    using parent_t::raw;

    /**
     * @brief Returns a sub-view into this view. Useful for working with slices.
     */
    auto subview(size_t begin, size_t count = std::dynamic_extent) noexcept -> vec_view<ValueType> {
        return vec_view{this->raw().subspan(begin, count)};
    }
    auto subview(size_t begin, size_t count = std::dynamic_extent) const noexcept -> vec_view<const ValueType> {
        return vec_view{this->raw().subspan(begin, count)};
    }
};

template<typename ValueType>
class vec : public vec_impl<ValueType, memory::cache_aligned_vector>
{
    using parent_t = vec_impl<ValueType, memory::cache_aligned_vector>;
    using container_t = typename parent_t::container_t;

public:
    explicit vec(size_t size, const ValueType& init = {}) : parent_t{container_t(size, init)} {}

    /**
     * @brief Converts an owning vector to a view to itself.
     */
    auto view() noexcept -> vec_view<ValueType> { return vec_view<ValueType>{this->raw()}; }
    auto view() const noexcept -> vec_view<const ValueType> { return vec_view<const ValueType>{this->raw()}; }

    /**
     * @brief Returns a sub-view into this vec. Useful for working with slices.
     */
    auto subview(auto&&... args) noexcept -> decltype(auto) {
        return view().subview(std::forward<decltype(args)>(args)...);
    }
    auto subview(auto&&... args) const noexcept -> decltype(auto) {
        return view().subview(std::forward<decltype(args)>(args)...);
    }
};

} // namespace dim

#endif /* INCLUDE_DIM_VEC */
