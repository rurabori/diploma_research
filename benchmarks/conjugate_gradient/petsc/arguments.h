#ifndef APPS_PETSC_BASELINE_ARGUMENTS
#define APPS_PETSC_BASELINE_ARGUMENTS

#include <array>

#include <petsc.h>

struct arguments
{
    std::array<char, PETSC_MAX_PATH_LEN> log_directory{'.'};
    std::array<char, PETSC_MAX_PATH_LEN> input_matrix{"input.mat"};
    std::array<char, PETSC_MAX_PATH_LEN> output_file{"out.h5"};
    std::array<char, 150> matrix_name{"A"};
    std::array<char, 150> result_name{"Y"};

    auto create() -> void;
};

#endif /* APPS_PETSC_BASELINE_ARGUMENTS */
