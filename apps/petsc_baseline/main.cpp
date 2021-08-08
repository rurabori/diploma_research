#include <exception>
#include <filesystem>
#include <fmt/core.h>
#include <iostream>

#include <petsc.h>
#include <petscoptions.h>
#include <petscsys.h>

#include <petscsystypes.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

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

    auto create() -> int {
        PetscBool found{PETSC_FALSE};

        auto ierr = PetscOptionsBegin(PETSC_COMM_WORLD, nullptr, "Options for this program", nullptr);

        ierr = PetscOptionsString("-log_directory", "directory in which to store logfiles, defaults to cwd", nullptr,
                                  std::data(log_directory), std::data(log_directory), std::size(log_directory), &found);
        CHKERRQ(ierr);

        ierr = PetscOptionsString("-matrix_file", "load the input matrix from the specified file", nullptr,
                                  std::data(input_matrix), std::data(input_matrix), std::size(input_matrix), &found);
        CHKERRQ(ierr);

        PetscOptionsEnd();

        return ierr;
    }
};

int main(int argc, char* argv[]) try {
    auto guard = petsc_guard(&argc, &argv, nullptr, help);

    arguments args{};
    args.create();

    auto logger = spdlog::basic_logger_mt(petsc_baseline_FULL_NAME, create_logger_name(std::data(args.log_directory)));

    try {
    } catch (std::exception& e) {
        logger->critical("failed with {}, aborting.", e.what());
        return 1;
    }

    return 0;
} catch (std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
}