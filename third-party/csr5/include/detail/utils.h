#ifndef UTILS_H
#define UTILS_H

#include "common.h"

#include <dirent.h>
#include <sys/time.h>
#include <sys/types.h>

#include <chrono>


namespace csr5::avx2 {

template<typename iT, typename vT>
double getB(const iT m, const iT nnz) {
    return (double)((m + 1 + nnz) * sizeof(iT) + (2 * nnz + m) * sizeof(vT));
}

template<typename iT>
double getFLOP(const iT nnz) {
    return (double)(2 * nnz);
}

template<typename T>
void print_tile_t(T* input, int m, int n) {
    for (int i = 0; i < n; i++) {
        for (int local_id = 0; local_id < m; local_id++) {
            cout << input[local_id * n + i] << ", ";
        }
        cout << endl;
    }
}

template<typename T>
void print_tile(T* input, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int local_id = 0; local_id < n; local_id++) {
            cout << input[i * n + local_id] << ", ";
        }
        cout << endl;
    }
}

template<typename T>
void print_1darray(T* input, int l) {
    for (int i = 0; i < l; i++)
        cout << input[i] << ", ";
    cout << endl;
}

template<typename Integral>
constexpr size_t get_needed_storage(Integral value) {
    size_t retval{1};

    // TODO: this might be doable with bit magic without a loop.
    for (Integral base = 2; base < value; base *= 2)
        ++retval;

    return retval;
}

template<typename Callable>
auto timed_section(Callable&& callable) {
    auto begin = std::chrono::steady_clock::now();
    std::forward<Callable>(callable)();
    return std::chrono::steady_clock::now() - begin;
}

} // namespace csr5::avx2

#endif // UTILS_H
