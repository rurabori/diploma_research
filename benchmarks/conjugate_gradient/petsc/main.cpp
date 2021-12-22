#include <dim/bench/stopwatch.h>
#include <dim/bench/timed_section.h>
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
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <nlohmann/json.hpp>

#include <exception>
#include <filesystem>
#include <fstream>
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

using second = dim::bench::second;
using dim::bench::section;
using dim::bench::stopwatch;

struct general_stats_t
{
    size_t local_rows;
    size_t local_elements;
    size_t node{dim::mpi::rank()};
    size_t node_count{dim::mpi::size()};
    size_t thread_count{::omp_get_max_threads()};
};

void to_json(nlohmann::json& j, const general_stats_t& stats) {
    j = nlohmann::json{
      {"local_rows", stats.local_rows}, {"local_elements", stats.local_elements}, {"node", stats.node},
      {"node_count", stats.node_count}, {"thread_count", stats.thread_count},
    };
}

struct cg_stats_t
{
    using second = dim::bench::second;

    size_t num_iters{};
    second total{};

    struct steps_t
    {
        second spmv{};
        second alpha{};
        second part_x{};
        second part_r{};
        second part_s{};
        second r_r{};
    } steps;
};

void to_json(nlohmann::json& j, const cg_stats_t& stats) {
    j = nlohmann::json{{"num_iters", stats.num_iters},
                       {"total", stats.total.count()},
                       {"steps",
                        {{"A*s", stats.steps.spmv.count()},
                         {"alpha = (r*r) / s*A*s", stats.steps.alpha.count()},
                         {"x += s * alpha", stats.steps.part_x.count()},
                         {"r -= temp * alpha", stats.steps.part_r.count()},
                         {"sk+1 = rk+1 + beta * sk;", stats.steps.part_s.count()},
                         {"r*r", stats.steps.r_r.count()}}}};
}
auto petsc_main(const arguments& args) -> void {
    using vec_t = guard<Vec, VecCreateMPI, VecDestroy>;

    const auto global_sw = stopwatch{};

    auto general_stats = general_stats_t{};

    auto local_sw = stopwatch{};
    auto A = load_partial(args);
    const auto io_time = local_sw.elapsed();

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

    auto r_r = vec_mul(r, r);

    auto cg_stats = cg_stats_t{};
    cg_stats.num_iters = 100;
    cg_stats.total = section([&] {
        for (size_t iter = 0; iter < 100; ++iter) {
            if (const auto norm_r = std::sqrt(r_r); norm_r / norm_b <= 1.e-8)
                break;

            cg_stats.steps.spmv += section([&] { petsc_try MatMult(A, s, temp); });

            local_sw.reset();
            const auto alpha = r_r / vec_mul(s, temp);
            cg_stats.steps.alpha += local_sw.elapsed();

            // x += alpha * s;
            cg_stats.steps.part_x += section([&] { petsc_try VecAXPY(x, alpha, s); });

            // r -= alpha * temp;
            cg_stats.steps.part_r += section([&] { petsc_try VecAXPY(r, -alpha, temp); });

            local_sw.reset();
            const auto tmp = vec_mul(r, r);
            const auto beta = tmp / std::exchange(r_r, tmp);
            cg_stats.steps.r_r += local_sw.elapsed();

            // sk+1 = rk+1 + beta * sk;
            cg_stats.steps.part_s += section([&] { petsc_try VecAYPX(s, beta, r); });
        }
    });
    const auto global_time = global_sw.elapsed();

    std::ofstream{fmt::format("stats_{:02}.json", dim::mpi::rank())} << std::setw(4)
                                                                     << nlohmann::json({
                                                                          {"total", global_time.count()},
                                                                          {"general_info", general_stats},
                                                                          {"cg", cg_stats},
                                                                          {"io", io_time.count()},
                                                                        });
}

int main(int argc, char* argv[]) try {
    auto guard = init_guard(&argc, &argv, nullptr, help);

    arguments args{};
    args.create();

    spdlog::set_default_logger(
      spdlog::stderr_logger_mt(fmt::format("{} (n{:02})", brr::app_info.full_name, dim::mpi::rank())));

    if (is_help_set())
        return 0;

    petsc_main(args);

    return 0;
} catch (std::exception& e) {
    spdlog::critical("failed with {}, aborting.", e.what());
    return 1;
}