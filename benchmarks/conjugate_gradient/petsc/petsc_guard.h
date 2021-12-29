#ifndef BENCHMARKS_CONJUGATE_GRADIENT_PETSC_PETSC_GUARD
#define BENCHMARKS_CONJUGATE_GRADIENT_PETSC_PETSC_GUARD

#include <petsc.h>

#include "petsc_error.h"

template<typename Ty, const auto& Creator, const auto& Deleter>
struct guard
{
    struct uninitialized_t
    {};

    struct retain_t
    {};

    Ty value{};

    template<typename... Args>
    explicit guard(Args&&... args) {
        petsc_try Creator(std::forward<Args>(args)..., &value);
    }

    explicit guard(Ty val, retain_t /*unused*/) : value{val} {}

    explicit guard(uninitialized_t /*unused*/) {}
    guard(const guard& other) = delete;
    guard(guard&& other) noexcept : value{std::exchange(other.value, nullptr)} {};

    guard& operator=(const guard& other) = delete;
    guard& operator=(guard&& other) noexcept { std::swap(other.value, value); }

    static guard uninitialized() { return guard{uninitialized_t{}}; }
    static guard retain(Ty val) { return guard{val, retain_t{}}; }

    ~guard() { release(); }

    auto release() -> void {
        if (!value)
            return;

        Deleter(&value);
        value = nullptr;
    }

    PetscObject as_object() { return reinterpret_cast<PetscObject>(value); }

    // NOLINTNEXTLINE - we want this to be convertible so that code is easy to read.
    operator Ty() { return value; }

    // NOLINTNEXTLINE - we want this to be convertible so that code is easy to read.
    operator PetscObject() { return as_object(); }

    auto value_ptr() -> Ty* { return &value; }
};

struct init_guard
{
    template<typename... Args>
    explicit init_guard(Args&&... args) {
        PetscInitialize(std::forward<Args>(args)...);
    }

    ~init_guard() { PetscFinalize(); }
};

#endif /* BENCHMARKS_CONJUGATE_GRADIENT_PETSC_PETSC_GUARD */
