#include <dim/mpi/mpi.h>

#include <petsc.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscsys.h>
#include <petscviewer.h>
#include <petscviewerhdf5.h>

#include <fmt/core.h>

#include <spdlog/logger.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <system_error>

#include "arguments.h"
#include "petsc_error.h"
#include "petsc_guard.h"
#include "version.h"

static constexpr const char* help = "Petsc baseline benchmark\n";

auto create_logger_name(const std::filesystem::path& base_dir) -> std::string {
    return base_dir / fmt::format("petsc_{:03d}.log", dim::mpi::rank());
}

auto is_help_set() -> bool {
    auto help_set = PetscBool{};
    petsc_try PetscOptionsHasName(nullptr, nullptr, "-help", &help_set);

    return help_set == PETSC_TRUE;
}

auto petsc_main(const arguments& args) -> void {
    using viewer_t = guard<PetscViewer, PetscViewerHDF5Open, PetscViewerDestroy>;
    using mat_t = guard<Mat, MatCreate, MatDestroy>;
    using vec_t = guard<Vec, VecCreate, VecDestroy>;

    mat_t A{PETSC_COMM_WORLD};
    petsc_try PetscObjectSetName(A, std::data(args.matrix_name));

    viewer_t in{PETSC_COMM_WORLD, std::data(args.input_matrix), FILE_MODE_READ};
    petsc_try MatLoad(A, in);

    auto X = vec_t::uninitialized();
    auto Y = vec_t::uninitialized();
    petsc_try MatCreateVecs(A, X.value_ptr(), Y.value_ptr());

    petsc_try PetscObjectSetName(Y, std::data(args.result_name));

    petsc_try VecSet(X, 1.0);

    petsc_try MatMult(A, X, Y);

    viewer_t out{PETSC_COMM_WORLD, std::data(args.output_file), FILE_MODE_WRITE};
    petsc_try VecView(Y, out);
}

int main(int argc, char* argv[]) try {
    auto guard = init_guard(&argc, &argv, nullptr, help);

    arguments args{};
    args.create();

    spdlog::set_default_logger(
      spdlog::basic_logger_mt(brr::app_info.full_name, create_logger_name(std::data(args.log_directory))));

    if (is_help_set())
        return 0;

    petsc_main(args);

    return 0;
} catch (std::exception& e) {
    spdlog::critical("failed with {}, aborting.", e.what());
    return 1;
}