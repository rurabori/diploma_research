#ifndef INCLUDE_DIM_KEYVAL_ITERATOR
#define INCLUDE_DIM_KEYVAL_ITERATOR

#include <algorithm>
#include <compare>
#include <concepts>
#include <ranges>

namespace dim {

template<typename Key, typename Value>
struct key_value_t
{
    Key key;
    Value val;

    friend auto operator<(const key_value_t& l, const key_value_t& r) { return l.key < r.key; }
};

template<typename KeyIt, typename ValueIt>
struct keyval_iterator_proxy_t
{
    using keyval_t = key_value_t<typename KeyIt::value_type, typename ValueIt::value_type>;

    KeyIt key_it;
    ValueIt val_it;

    keyval_iterator_proxy_t(KeyIt key, ValueIt val) : key_it{key}, val_it{val} {}
    keyval_iterator_proxy_t(const keyval_iterator_proxy_t& r) noexcept = default;
    keyval_iterator_proxy_t(keyval_iterator_proxy_t&& r) noexcept = default;
    ~keyval_iterator_proxy_t() noexcept = default;

    keyval_iterator_proxy_t& operator=(const keyval_iterator_proxy_t& r) noexcept {
        if (this == &r)
            return *this;

        *key_it = *r.key_it;
        *val_it = *r.val_it;
        return *this;
    }

    keyval_iterator_proxy_t& operator=(keyval_iterator_proxy_t&& r) noexcept {
        *key_it = std::move(*r.key_it);
        *val_it = std::move(*r.val_it);
        return *this;
    }

    keyval_iterator_proxy_t& operator=(keyval_t&& r) noexcept {
        *key_it = std::move(r.key);
        *val_it = std::move(r.val);
        return *this;
    }

    explicit(false) operator keyval_t() && { return keyval_t{std::move(*key_it), std::move(*val_it)}; }

    friend auto operator<(const keyval_t& l, const keyval_iterator_proxy_t& r) { return l.key < *r.key_it; }
    friend auto operator<(const keyval_iterator_proxy_t& l, const keyval_t& r) { return *l.key_it < r.key; }
    friend auto operator<(const keyval_iterator_proxy_t& l, const keyval_iterator_proxy_t& r) {
        return *l.key_it < *r.key_it;
    }

    friend void swap(keyval_iterator_proxy_t l, keyval_iterator_proxy_t r) {
        using std::swap;
        swap(*l.key_it, *r.key_it);
        swap(*l.val_it, *r.val_it);
    }
};

template<typename KeyIt, typename ValueIt>
struct keyval_iterator_t
{
    using proxy_t = keyval_iterator_proxy_t<KeyIt, ValueIt>;

    using value_type = key_value_t<typename KeyIt::value_type, typename ValueIt::value_type>;
    using reference_type = proxy_t;

    KeyIt key_it;
    ValueIt val_it;

    auto operator-(const keyval_iterator_t& r) const { return key_it - r.key_it; }
    auto operator+(std::integral auto off) const { return keyval_iterator_t{key_it + off, val_it + off}; }
    auto operator-(std::integral auto off) const { return keyval_iterator_t{key_it - off, val_it - off}; }
    keyval_iterator_t& operator++() {
        ++key_it;
        ++val_it;
        return *this;
    }

    keyval_iterator_t& operator--() {
        --key_it;
        --val_it;
        return *this;
    }

    auto operator++(int) { return keyval_iterator_t{key_it++, val_it++}; }
    auto operator--(int) { return keyval_iterator_t{key_it--, val_it--}; }
    value_type operator*() const { return value_type{*key_it, *val_it}; }
    reference_type operator*() { return proxy_t{key_it, val_it}; }

    friend auto operator==(const keyval_iterator_t& l, const keyval_iterator_t& r) noexcept {
        return l.key_it == r.key_it;
    }
    friend auto operator!=(const keyval_iterator_t& l, const keyval_iterator_t& r) noexcept {
        return l.key_it != r.key_it;
    }

    friend auto operator<(const keyval_iterator_t& l, const keyval_iterator_t& r) noexcept {
        return l.key_it < r.key_it;
    }
};

void keyval_sort(auto&& keys, auto&& values) {
    std::sort(keyval_iterator_t{keys.begin(), values.begin()}, keyval_iterator_t{keys.end(), values.end()});
}

} // namespace dim

#endif /* INCLUDE_DIM_KEYVAL_ITERATOR */
