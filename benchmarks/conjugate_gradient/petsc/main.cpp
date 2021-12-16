#include <dim/mpi/mpi.h>

#include <mpi.h>
#include <petsc.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscviewer.h>

#include <fmt/core.h>

#include <spdlog/logger.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <system_error>

#include <dim/io/h5.h>

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
    using viewer_t = guard<PetscViewer, PetscViewerBinaryOpen, PetscViewerDestroy>;
    using mat_t = guard<Mat, MatCreateMPIAIJWithArrays, MatDestroy>;
    using vec_t = guard<Vec, VecCreate, VecDestroy>;

    auto csr = dim::io::h5::read_matlab_compatible(args.input_matrix.data(), args.matrix_name.data());

    auto A = mat_t{MPI_COMM_WORLD,
                   csr.dimensions.rows,
                   csr.dimensions.cols,
                   csr.dimensions.rows,
                   csr.dimensions.cols,
                   reinterpret_cast<PetscInt*>(csr.row_start_offsets.data()),
                   reinterpret_cast<PetscInt*>(csr.col_indices.data()),
                   csr.values.data()};

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