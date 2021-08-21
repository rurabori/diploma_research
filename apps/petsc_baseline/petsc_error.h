#ifndef APPS_PETSC_BASELINE_PETSC_ERROR
#define APPS_PETSC_BASELINE_PETSC_ERROR

#include <stdexcept>

#include <petsc.h>

struct petsc_error_checker
{
    const char* file;
    const int line;
    const char* function;

    [[nodiscard]] int contain_errq(int errc) const {
        if (errc != 0)
            return PetscError(PETSC_COMM_SELF, line, function, file, errc, PETSC_ERROR_REPEAT, " ");

        return 0;
    }

    void operator%(int errc) const {
        if (contain_errq(errc) != 0)
            throw std::system_error{errc, std::generic_category(), "petsc failed."};
    }
};

// NOLINTNEXTLINE - this is the cleanest way to avoid petsc handling control flow too much.
#define petsc_try petsc_error_checker{__FILE__, __LINE__, PETSC_FUNCTION_NAME} %

#endif /* APPS_PETSC_BASELINE_PETSC_ERROR */
