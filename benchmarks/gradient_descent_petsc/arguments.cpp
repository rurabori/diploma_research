#include "arguments.h"

#include "petsc_error.h"

auto arguments::create() -> void {
    petsc_try[&] {
        PetscBool found{PETSC_FALSE};
        petsc_try PetscOptionsBegin(PETSC_COMM_WORLD, nullptr, "Options for this program", nullptr);

        petsc_try PetscOptionsString("-log_directory", "directory in which to store logfiles, defaults to cwd", nullptr,
                                     std::data(log_directory), std::data(log_directory), std::size(log_directory),
                                     &found);

        petsc_try PetscOptionsString("-matrix_file", "load the input matrix from the specified file", nullptr,
                                     std::data(input_matrix), std::data(input_matrix), std::size(input_matrix), &found);

        petsc_try PetscOptionsString("-result_file", "store the result in this file", nullptr, std::data(output_file),
                                     std::data(output_file), std::size(output_file), &found);

        petsc_try PetscOptionsString("-matrix_name", "name of the matrix to load (names group in HDF5 file)", nullptr,
                                     std::data(matrix_name), std::data(matrix_name), std::size(matrix_name), &found);

        petsc_try PetscOptionsString("-result_name", "name of result vector (names group in HDF5 file)", nullptr,
                                     std::data(result_name), std::data(result_name), std::size(result_name), &found);

        PetscOptionsEnd();
        return 0;
    }
    ();
}
