#include <dim/io/h5.h>
#include <dim/mpi/csr.h>
#include <dim/mpi/mpi.h>

#include <mpi.h>
#include <petsc.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>

#include <fmt/chrono.h>
#include <fmt/core.h>

#include <spdlog/logger.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

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

template<typename Result, const auto& Fun, typename... Args>
auto petsc_call(Args&&... args) -> Result {
    Result result{};
    petsc_try Fun(std::forward<Args>(args)..., &result);
    return result;
}

auto load_partial(const arguments& args) {
    using mat_t = guard<Mat, MatCreateMPIAIJWithArrays, MatDestroy>;
    auto partial = dim::mpi::load_csr_partial(args.input_matrix.data(), args.matrix_name.data(), PETSC_COMM_WORLD);

    return mat_t{PETSC_COMM_WORLD,
                 static_cast<int>(partial.matrix_chunk.dimensions.rows),
                 PETSC_DECIDE,
                 static_cast<int>(partial.global_dimensions.rows),
                 static_cast<int>(partial.global_dimensions.cols),
                 reinterpret_cast<PetscInt*>(partial.matrix_chunk.row_start_offsets.data()),
                 reinterpret_cast<PetscInt*>(partial.matrix_chunk.col_indices.data()),
                 partial.matrix_chunk.values.data()};
}

auto vec_norm(Vec vec) -> PetscReal { return petsc_call<PetscReal, VecNorm>(vec, NORM_2); }
auto vec_mul(Vec lhs, Vec rhs) -> PetscReal { return petsc_call<PetscScalar, VecDot>(lhs, rhs); }

auto petsc_main(const arguments& args) -> void {
    using vec_t = guard<Vec, VecCreateMPI, VecDestroy>;

    auto A = load_partial(args);

    auto s = vec_t::uninitialized();
    auto temp = vec_t::uninitialized();
    petsc_try MatCreateVecs(A, s.value_ptr(), temp.value_ptr());

    // s = rhs
    petsc_try VecSet(s, 1.0);

    // r = s
    auto size = petsc_call<PetscInt, VecGetSize>(s);
    auto r = vec_t{PETSC_COMM_WORLD, PETSC_DECIDE, size};
    petsc_try VecCopy(s, r);

    auto x = vec_t{PETSC_COMM_WORLD, PETSC_DECIDE, size};
    petsc_try VecSet(x, 0.);

    const auto norm_b = vec_norm(s);
    spdlog::info("norm b = {}", norm_b);

    auto r_r = vec_mul(r, r);

    spdlog::stopwatch global_sw;
    for (size_t iter = 0; iter < 100; ++iter) {
        if (const auto norm_r = vec_norm(r); norm_r / norm_b <= 1.e-8)
            break;

        spdlog::stopwatch sw;
        petsc_try MatMult(A, s, temp);
        spdlog::info("spmv took {}", sw.elapsed());

        sw.reset();
        const auto alpha = r_r / vec_mul(s, temp);
        spdlog::info("alpha computation took {}", sw.elapsed());

        // x += alpha * s;
        VecAXPY(x, alpha, s);

        // r -= alpha * temp;
        VecAXPY(r, -alpha, temp);

        const auto tmp = vec_mul(r, r);
        const auto beta = tmp / std::exchange(r_r, tmp);

        sw.reset();
        // sk+1 = rk+1 + beta * sk;
        VecAYPX(s, beta, r);
        spdlog::info("s computation took {}", sw.elapsed());
    }
    spdlog::info("100 iterations done in {}", global_sw.elapsed());
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