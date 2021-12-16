#ifndef APPS_PETSC_BASELINE_PETSC_GUARD
#define APPS_PETSC_BASELINE_PETSC_GUARD

#include <petsc.h>

#include "petsc_error.h"

template<typename Ty, const auto& Creator, const auto& Deleter>
struct guard
{
    struct uninitialized_t
    {};

    Ty value;

    template<typename... Args>
    explicit guard(Args&&... args) {
        petsc_try Creator(std::forward<Args>(args)..., &value);
    }

    explicit guard(uninitialized_t /*unused*/) {}

    static guard uninitialized() { return guard{uninitialized_t{}}; }

    ~guard() { Deleter(&value); }

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

#endif /* APPS_PETSC_BASELINE_PETSC_GUARD */
