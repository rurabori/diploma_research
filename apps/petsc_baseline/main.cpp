#include <fmt/core.h>

#include <petsc.h>
#include <petscerror.h>
#include <petscoptions.h>
#include <petscsys.h>
#include <petscviewerhdf5.h>

#include <petscsystypes.h>
#include <petscviewer.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <system_error>

#include "version.h"

static constexpr const char* help = "Petsc baseline benchmark" petsc_baseline_VER "\n";

struct petsc_guard
{
    template<typename... Args>
    explicit petsc_guard(Args&&... args) {
        PetscInitialize(std::forward<Args>(args)...);
    }

    ~petsc_guard() { PetscFinalize(); }
};

struct petsc_error_checker
{
    const char* file;
    const int line;
    const char* function;

    [[nodiscard]] int contain_errq(int errc) const {
        if (errc != 0) [[unlikely]]
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

int get_rank() {
    int rank{};
    ::MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    return rank;
}

auto create_logger_name(const std::filesystem::path& base_dir) -> std::string {
    return base_dir / fmt::format("petsc_{:03d}.log", get_rank());
}

struct arguments
{
    std::array<char, PETSC_MAX_PATH_LEN> log_directory{'.'};
    std::array<char, PETSC_MAX_PATH_LEN> input_matrix{"input.mat"};
    std::array<char, 150> matrix_name{"A"};

    auto create() -> void {
        petsc_try[&] {
            PetscBool found{PETSC_FALSE};
            petsc_try PetscOptionsBegin(PETSC_COMM_WORLD, nullptr, "Options for this program", nullptr);

            petsc_try PetscOptionsString("-log_directory", "directory in which to store logfiles, defaults to cwd",
                                         nullptr, std::data(log_directory), std::data(log_directory),
                                         std::size(log_directory), &found);

            petsc_try PetscOptionsString("-matrix_file", "load the input matrix from the specified file", nullptr,
                                         std::data(input_matrix), std::data(input_matrix), std::size(input_matrix),
                                         &found);

            petsc_try PetscOptionsString("-matrix_name", "name of the matrix to load (names group in HDF5 file)",
                                         nullptr, std::data(matrix_name), std::data(matrix_name),
                                         std::size(matrix_name), &found);

            PetscOptionsEnd();
            return 0;
        }
        ();
    }
};

auto is_help_set() -> bool {
    auto help_set = PetscBool{};
    petsc_try PetscOptionsHasName(nullptr, nullptr, "-help", &help_set);

    return help_set == PETSC_TRUE;
}

auto petsc_main(const arguments& args, std::shared_ptr<spdlog::logger> logger) -> void {
    PetscViewer in{};
    petsc_try PetscViewerHDF5Open(MPI_COMM_WORLD, std::data(args.input_matrix), FILE_MODE_READ, &in);

    Mat A{};
    petsc_try MatCreate(PETSC_COMM_WORLD, &A);
    petsc_try PetscObjectSetName(reinterpret_cast<PetscObject>(A), std::data(args.matrix_name));
    petsc_try MatLoad(A, in);
    petsc_try MatView(A, PETSC_VIEWER_STDOUT_SELF);
    petsc_try MatDestroy(&A);
    petsc_try PetscViewerDestroy(&in);
}

int main(int argc, char* argv[]) try {
    auto guard = petsc_guard(&argc, &argv, nullptr, help);

    arguments args{};
    args.create();

    auto logger = spdlog::basic_logger_mt(petsc_baseline_FULL_NAME, create_logger_name(std::data(args.log_directory)));

    if (is_help_set())
        return 0;

    try {
        petsc_main(args, logger);
    } catch (std::exception& e) {
        logger->critical("failed with {}, aborting.", e.what());
        return 1;
    }

    return 0;
} catch (std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
}