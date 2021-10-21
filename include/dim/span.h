#ifndef INCLUDE_DIM_SPAN
#define INCLUDE_DIM_SPAN

// This file was taken from Hana Dusíková's LUNA library.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <type_traits>

#if !__has_include(<span>)
namespace luna {

static constexpr size_t dynamic_extent = ~static_cast<size_t>(0);

template<typename T, size_t Extent = dynamic_extent>
class span;

// definition of internal types

template<typename T>
struct _span_types
{
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using index_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
};

// difference for dynamic_extent and empty span

template<typename T, size_t Extent>
struct _basic_span : public _span_types<T>
{
    using _types = _span_types<T>;

    using element_type = typename _types::element_type;
    using value_type = typename _types::value_type;
    using index_type = typename _types::index_type;
    using difference_type = typename _types::difference_type;
    using pointer = typename _types::pointer;
    using const_pointer = typename _types::const_pointer;
    using reference = typename _types::reference;
    using const_reference = typename _types::const_reference;
    using iterator = typename _types::iterator;
    using const_iterator = typename _types::const_iterator;
    using reverse_iterator = typename _types::reverse_iterator;
    using const_reverse_iterator = typename _types::const_reverse_iterator;

    pointer _pointer_to_data;

    constexpr _basic_span() noexcept = delete; // you can't default initialize fixed sized span
    constexpr _basic_span(pointer ptr, [[maybe_unused]] size_t count) noexcept : _pointer_to_data{ptr} {
        assert(count == Extent);
    }

    constexpr _basic_span(const _basic_span& other) noexcept = default;

    constexpr _basic_span& operator=(const _basic_span& other) noexcept = default;

    [[nodiscard]] constexpr pointer data() const noexcept { return _pointer_to_data; }

    [[nodiscard]] constexpr size_t size() const noexcept { return Extent; }
};

// empty span (special case)

template<typename T>
struct _basic_span<T, 0> : public _span_types<T>
{
    using _types = _span_types<T>;

    using element_type = typename _types::element_type;
    using value_type = typename _types::value_type;
    using index_type = typename _types::index_type;
    using difference_type = typename _types::difference_type;
    using pointer = typename _types::pointer;
    using const_pointer = typename _types::const_pointer;
    using reference = typename _types::reference;
    using const_reference = typename _types::const_reference;
    using iterator = typename _types::iterator;
    using const_iterator = typename _types::const_iterator;
    using reverse_iterator = typename _types::reverse_iterator;
    using const_reverse_iterator = typename _types::const_reverse_iterator;

    constexpr _basic_span() noexcept = default;
    constexpr _basic_span(pointer, [[maybe_unused]] size_t count) noexcept { assert(count == 0); }

    constexpr _basic_span(const _basic_span& other) noexcept = default;

    constexpr _basic_span& operator=(const _basic_span& other) noexcept = default;

    [[nodiscard]] constexpr pointer data() const noexcept { return nullptr; }

    [[nodiscard]] constexpr size_t size() const noexcept { return 0; }
};

// dynamically sized span

template<typename T>
struct _basic_span<T, dynamic_extent> : public _span_types<T>
{
    using _types = _span_types<T>;

    using element_type = typename _types::element_type;
    using value_type = typename _types::value_type;
    using index_type = typename _types::index_type;
    using difference_type = typename _types::difference_type;
    using pointer = typename _types::pointer;
    using const_pointer = typename _types::const_pointer;
    using reference = typename _types::reference;
    using const_reference = typename _types::const_reference;
    using iterator = typename _types::iterator;
    using const_iterator = typename _types::const_iterator;
    using reverse_iterator = typename _types::reverse_iterator;
    using const_reverse_iterator = typename _types::const_reverse_iterator;

    pointer _pointer_to_data{nullptr};
    index_type _number_of_objects{0};

    constexpr _basic_span() noexcept = default;
    constexpr _basic_span(pointer ptr, index_type count) noexcept : _pointer_to_data{ptr}, _number_of_objects{count} {}

    constexpr _basic_span(const _basic_span& other) noexcept = default;

    constexpr _basic_span& operator=(const _basic_span& other) noexcept = default;

    [[nodiscard]] constexpr pointer data() const noexcept { return _pointer_to_data; }

    [[nodiscard]] constexpr size_t size() const noexcept { return _number_of_objects; }
};

namespace impl {

    template<typename T>
    struct _valid_span_container
    { constexpr static bool value = !std::is_array_v<T>; };

    template<typename T, size_t N>
    struct _valid_span_container<luna::span<T, N>>
    { constexpr static bool value = false; };

    template<typename T, size_t N>
    struct _valid_span_container<std::array<T, N>>
    { constexpr static bool value = false; };

    template<typename T>
    struct _valid_data_size_for_container
    {
        template<typename Y>
        static auto test(Y*)
          -> decltype(std::data(std::declval<Y&>()), std::size(std::declval<Y&>()), std::true_type{});
        template<typename>
        static auto test(...) -> std::false_type;

        constexpr static bool value = decltype(test<T>(0))::value;
    };

} // namespace impl

template<typename T>
static constexpr bool _valid_span_container_v
  = impl::_valid_span_container<T>::value&& impl::_valid_data_size_for_container<T>::value;

template<typename T, size_t Extent>
class span : protected _basic_span<T, Extent>
{
    using _basic_span_impl = _basic_span<T, Extent>;

public:
    using element_type = typename _basic_span_impl::element_type;
    using value_type = typename _basic_span_impl::value_type;
    using index_type = typename _basic_span_impl::index_type;
    using difference_type = typename _basic_span_impl::difference_type;
    using pointer = typename _basic_span_impl::pointer;
    using const_pointer = typename _basic_span_impl::const_pointer;
    using reference = typename _basic_span_impl::reference;
    using const_reference = typename _basic_span_impl::const_reference;
    using iterator = typename _basic_span_impl::iterator;
    using const_iterator = typename _basic_span_impl::const_iterator;
    using reverse_iterator = typename _basic_span_impl::reverse_iterator;
    using const_reverse_iterator = typename _basic_span_impl::const_reverse_iterator;

    // constructors
    constexpr span() noexcept : _basic_span_impl{} {}
    constexpr span(pointer ptr, size_t count) noexcept : _basic_span_impl{ptr, count} {}
    constexpr span(pointer first, pointer last) noexcept
      : _basic_span_impl{first, static_cast<index_type>(std::distance(first, last))} {}

    template<size_t N>
    constexpr span(element_type (&arr)[N]) noexcept : _basic_span_impl{arr, N} {}

    template<size_t N>
    constexpr span(std::array<value_type, N>& arr) noexcept : _basic_span_impl{arr.data(), N} {}
    template<size_t N>
    constexpr span(const std::array<value_type, N>& arr) noexcept : _basic_span_impl{arr.data(), N} {}

    template<
      typename Container,
      typename = std::enable_if_t<
        _valid_span_container_v<
          Container> && std::is_convertible_v<std::remove_pointer_t<decltype(std::data(std::declval<Container&>()))> (*)[], element_type (*)[]>>>
    constexpr span(Container& cont) : _basic_span_impl{std::data(cont), std::size(cont)} {
        assert(Extent == luna::dynamic_extent || (Extent != luna::dynamic_extent && std::size(cont) != Extent));
    }
    template<
      typename Container,
      typename = std::enable_if_t<
        _valid_span_container_v<
          Container> && std::is_convertible_v<std::remove_pointer_t<decltype(std::data(std::declval<Container&>()))> (*)[], element_type (*)[]>>>
    constexpr span(const Container& cont) : _basic_span_impl{std::data(cont), std::size(cont)} {
        assert(Extent == luna::dynamic_extent || (Extent != luna::dynamic_extent && std::size(cont) != Extent));
    }

    template<typename U, size_t N,
             typename = std::enable_if_t<(Extent == luna::dynamic_extent || N == Extent)
                                         && std::is_convertible_v<U (*)[], element_type (*)[]>>>
    constexpr span(const luna::span<U, N>& s) noexcept : _basic_span_impl{s.data(), s.size()} {}

    constexpr span(const span& other) noexcept = default;

    constexpr span& operator=(const span& other) noexcept = default;

    [[nodiscard]] constexpr size_t size() const noexcept { return _basic_span_impl::size(); }
    [[nodiscard]] constexpr size_t size_bytes() const noexcept { return size() * sizeof(element_type); }
    [[nodiscard]] constexpr bool empty() const noexcept { return size() == 0; }

    constexpr reference front() const { return data()[0]; }
    constexpr reference back() const { return data()[size() - 1]; }
    constexpr reference operator[](index_type idx) const {
        assert(idx < size());
        return data()[idx];
    }
    constexpr pointer data() const noexcept { return _basic_span_impl::data(); }

    template<size_t Count>
    constexpr span<T, Count> first() const {
        assert(Count <= size());
        return span<T, Count>{data(), Count};
    }
    constexpr span<T, dynamic_extent> first(size_t count) const {
        assert(count <= size());
        return span<T, dynamic_extent>{data(), count};
    }

    template<size_t Count>
    constexpr span<T, Count> last() const {
        assert(Count <= size());
        return span<element_type, Count>{data() + (size() - Count), Count};
    }
    constexpr span<T, dynamic_extent> last(size_t count) const {
        assert(count <= size());
        return span<element_type, dynamic_extent>{data() + (size() - count), count};
    }

    template<size_t Offset, size_t Count = luna::dynamic_extent>
    constexpr auto subspan() const {
        assert(Offset <= size());
        if constexpr (Count != luna::dynamic_extent) {
            assert((Offset + Count) <= size());
            return span<element_type, Count>{data() + Offset, Count};
        } else if constexpr (Extent != luna::dynamic_extent) {
            return span<element_type, Extent - Offset>{data() + Offset, Extent - Offset};
        } else {
            return span<element_type, luna::dynamic_extent>{data() + Offset, size() - Offset};
        }
    }
    constexpr span<element_type, luna::dynamic_extent> subspan(size_t offset,
                                                               size_t count = luna::dynamic_extent) const {
        assert(offset <= size());
        if (count == luna::dynamic_extent) {
            return span<element_type, luna::dynamic_extent>{data() + offset, size() - offset};
        } else {
            assert(count != luna::dynamic_extent && (offset + count) <= size());
            return span<element_type, luna::dynamic_extent>{data() + offset, count};
        }
    }

    constexpr auto begin() const noexcept { return data(); }
    constexpr auto cbegin() const noexcept { return data(); }
    constexpr auto end() const noexcept { return data() + size(); }
    constexpr auto cend() const noexcept { return data() + size(); }
    constexpr auto rbegin() const noexcept { return std::make_reverse_iterator(end()); }
    constexpr auto crbegin() const noexcept { return std::make_reverse_iterator(cend()); }
    constexpr auto rend() const noexcept { return std::make_reverse_iterator(begin()); }
    constexpr auto crend() const noexcept { return std::make_reverse_iterator(cbegin()); }
};

// deduction rules
template<class T, class SzTy>
span(T*, SzTy) -> span<T>;
template<class T, std::size_t N>
span(T (&)[N]) -> span<T, N>;
template<class T, std::size_t N>
span(std::array<T, N>&) -> span<T, N>;
template<class T, std::size_t N>
span(const std::array<T, N>&) -> span<const T, N>;
template<class Container>
span(Container&) -> span<typename Container::value_type>;
template<class Container>
span(const Container&) -> span<const typename Container::value_type>;

// structured bindings and get<N>(span)

template<std::size_t I, class T, std::size_t N>
constexpr T& get(luna::span<T, N> s) noexcept {
    static_assert(N != luna::dynamic_extent, "Dynamic span is not supported!");
    return s[I];
}

// begin/end free functions

template<typename T, size_t N>
constexpr auto begin(luna::span<T, N> s) noexcept {
    return s.begin();
}

template<typename T, size_t N>
constexpr auto end(luna::span<T, N> s) noexcept {
    return s.end();
}

template<typename T, size_t N>
auto as_bytes(luna::span<T, N> s) noexcept {
    if constexpr (N == luna::dynamic_extent) {
        return luna::span<const uint8_t, luna::dynamic_extent>{reinterpret_cast<const uint8_t*>(s.data()),
                                                               s.size_bytes()};
    } else {
        return luna::span<const uint8_t, s.size_bytes()>{reinterpret_cast<const uint8_t*>(s.data()), s.size_bytes()};
    }
}

template<typename T, size_t N, typename = std::enable_if_t<!std::is_const_v<T>>>
auto as_writable_bytes(luna::span<T, N> s) noexcept {
    if constexpr (N == luna::dynamic_extent) {
        return luna::span<uint8_t, luna::dynamic_extent>{reinterpret_cast<uint8_t*>(s.data()), s.size_bytes()};
    } else {
        return luna::span<uint8_t, s.size_bytes()>{reinterpret_cast<uint8_t*>(s.data()), s.size_bytes()};
    }
}

template<typename T>
struct _is_span : std::false_type
{};

template<typename T, size_t Extent>
struct _is_span<span<T, Extent>> : std::true_type
{};

template<typename T>
inline constexpr bool is_span_v = _is_span<T>::value;

template<typename T, typename R>
auto operator==(span<T> lhs, R&& rhs) -> std::enable_if_t<std::is_convertible_v<R, span<T>>, bool> {
    span<T> rhs_span = rhs;
    return std::equal(lhs.begin(), lhs.end(), rhs_span.begin(), rhs_span.end());
}

template<typename L, typename T>
auto operator==(L&& lhs, span<T> rhs)
  -> std::enable_if_t<!is_span_v<std::decay_t<L>> && std::is_convertible_v<L, span<T>>, bool> {
    span<T> rhs_span = rhs;
    return std::equal(lhs.begin(), lhs.end(), rhs_span.begin(), rhs_span.end());
}

template<typename T, typename R>
auto operator!=(span<T> lhs, R&& rhs) -> std::enable_if_t<std::is_convertible_v<R, span<T>>, bool> {
    return !(lhs == rhs);
}

template<typename L, typename T>
auto operator!=(L&& lhs, span<T> rhs)
  -> std::enable_if_t<!is_span_v<std::decay_t<L>> && std::is_convertible_v<L, span<T>>, bool> {
    return !(lhs == rhs);
}

template<typename T, typename R>
auto operator<(span<T> lhs, R&& rhs) -> std::enable_if_t<std::is_convertible_v<R, span<T>>, bool> {
    span<T> rhs_span = rhs;
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs_span.begin(), rhs_span.end());
}

template<typename L, typename T>
auto operator<(L&& lhs, span<T> rhs)
  -> std::enable_if_t<!is_span_v<std::decay_t<L>> && std::is_convertible_v<L, span<T>>, bool> {
    return span<T>{lhs} < rhs;
}

template<typename T, typename R>
auto operator>(span<T> lhs, R&& rhs) -> std::enable_if_t<std::is_convertible_v<R, span<T>>, bool> {
    return rhs < lhs;
}

template<typename L, typename T>
auto operator>(L&& lhs, span<T> rhs)
  -> std::enable_if_t<!is_span_v<std::decay_t<L>> && std::is_convertible_v<L, span<T>>, bool> {
    return rhs < lhs;
}

template<typename T, typename R>
auto operator<=(span<T> lhs, R&& rhs) -> std::enable_if_t<std::is_convertible_v<R, span<T>>, bool> {
    return !(rhs < lhs);
}

template<typename L, typename T>
auto operator<=(L&& lhs, span<T> rhs)
  -> std::enable_if_t<!is_span_v<std::decay_t<L>> && std::is_convertible_v<L, span<T>>, bool> {
    return !(rhs < lhs);
}

template<typename T, typename R>
auto operator>=(span<T> lhs, R&& rhs) -> std::enable_if_t<std::is_convertible_v<R, span<T>>, bool> {
    return !(lhs < rhs);
}

template<typename L, typename T>
auto operator>=(L&& lhs, span<T> rhs)
  -> std::enable_if_t<!is_span_v<std::decay_t<L>> && std::is_convertible_v<L, span<T>>, bool> {
    return !(lhs < rhs);
}

} // namespace luna

// support for structured bindings

namespace std {

template<typename T, std::size_t N>
class tuple_size<luna::span<T, N>> : public std::integral_constant<std::size_t, N>
{};
template<typename T>
class tuple_size<luna::span<T, luna::dynamic_extent>>; // undefined

template<std::size_t I, typename T, std::size_t N>
struct tuple_element<I, luna::span<T, N>>
{ using type = T; };

} // namespace std

namespace dim {
using luna::as_bytes;
using luna::as_writable_bytes;
using luna::span;

} // namespace dim
#else
#include <span>
namespace dim {
using std::as_bytes;
using std::as_writable_bytes;
using std::span;
} // namespace dim

#endif
#endif /* INCLUDE_DIM_SPAN */
