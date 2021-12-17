#include <dim/csr5_mpi/csr5_mpi.h>
#include <dim/mpi/mpi.h>

#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <fmt/chrono.h>

#include <structopt/app.hpp>

#include <execution>
#include <filesystem>
#include <optional>
#include <string>

#include "version.h"

struct arguments_t
{
    std::filesystem::path input_file;
    std::optional<std::string> group_name{"A"};
    std::optional<double> threshold{0.1};
    std::optional<size_t> max_iters{100};
};
STRUCTOPT(arguments_t, input_file);

auto mpi_reduce(std::span<const double> a, std::span<const double> b) -> double {
    // we now have a * b partial result.
    const auto partial = std::transform_reduce(std::execution::par_unseq, a.begin(), a.end(), b.begin(), 0.);

    auto result = double{};
    ::MPI_Allreduce(&partial, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return result;
}

auto mpi_reduce(std::span<const double> a) -> double { return mpi_reduce(a, a); }

auto mpi_magnitude(std::span<const double> b) -> double { return std::sqrt(mpi_reduce(b, b)); }

constexpr auto op_equals = [](auto&& lhs, auto&& rhs, auto&& alpha, auto&& op) {
    std::transform(std::execution::par_unseq, std::begin(lhs), std::end(lhs), std::begin(rhs), std::begin(lhs),
                   [alpha, op](auto&& lhs_elem, auto&& rhs_elem) { return op(lhs_elem, alpha * rhs_elem); });
};

auto main_impl(const arguments_t& args) -> int {
    using dim::csr5_mpi::csr5_partial;
    constexpr auto csr5_strat = csr5_partial::matrix_type::spmv_strategy::partial;

    spdlog::stopwatch sw;

    const auto mpi_mat = csr5_partial::load(args.input_file, *args.group_name);
    spdlog::info("loading matrix took {}", sw.elapsed());

    const auto [first_row, last_row] = mpi_mat.output_range();
    const auto sync_first_row = mpi_mat.matrix.first_tile_uncapped();

    auto [edge_sync, result_sync] = mpi_mat.make_sync();

    // load the relevant part of b.
    // r0 = b - A*x0;  x0 = {0...} => r0 = b
    auto r = dim::mat::cache_aligned_vector<double>(mpi_mat.matrix.dimensions.cols, 1.);
    // s0 = r0
    auto s = dim::mat::cache_aligned_vector<double>(mpi_mat.matrix.dimensions.cols, 1.);

    const auto element_count = last_row - first_row + 1;

    // the part of b this matrix will be using.
    const auto owned_r = std::span{r}.subspan(first_row, element_count).subspan(sync_first_row);
    const auto owned_s = std::span{s}.subspan(first_row, element_count).subspan(sync_first_row);

    // norm_b = b.magnitude()
    const auto norm_b = mpi_magnitude(owned_r);
    spdlog::info("norm b = {}", norm_b);

    // the output vector for spmv.
    auto temp = dim::mat::cache_aligned_vector<double>(element_count, 0.);
    // and a view into it conditionally skipping first element.
    const auto owned_temp = std::span{temp}.subspan(sync_first_row);

    auto calibrator = mpi_mat.matrix.allocate_calibrator();

    auto x_partial = dim::mat::cache_aligned_vector<double>(owned_r.size(), 0.);

    auto r_r = mpi_reduce(owned_r);

    spdlog::stopwatch global_sw;
    for (size_t i = 0; i < *args.max_iters; ++i) {
        // end condition.
        if (mpi_magnitude(owned_r) / norm_b <= *args.threshold)
            break;

        sw.reset();
        // temp = A*s
        mpi_mat.matrix.spmv<csr5_strat>({.x = s, .y = temp, .calibrator = calibrator});
        spdlog::info("spmv took {}", sw.elapsed());

        sw.reset();
        edge_sync.sync(temp);
        spdlog::info("edge sync took {}", sw.elapsed());

        sw.reset();
        // alpha = (r * r)  / (s * temp)
        const auto alpha = r_r / mpi_reduce(owned_s, owned_temp);
        spdlog::info("alpha computation took {}", sw.elapsed());

        // x += s * alpha
        op_equals(x_partial, owned_s, alpha, std::plus<>{});

        // r -= temp * alpha
        op_equals(owned_r, owned_temp, alpha, std::minus<>{});

        auto tmp = mpi_reduce(owned_r);
        auto beta = tmp / std::exchange(r_r, tmp);

        sw.reset();
        // sk+1 = rk+1 + beta * sk;
        std::transform(std::execution::par_unseq, std::begin(owned_r), std::end(owned_r), std::begin(owned_s),
                       std::begin(owned_s), [beta](auto&& r, auto&& s) { return r + beta * s; });
        spdlog::info("computing partial s took {}", sw.elapsed());

        sw.reset();
        result_sync.sync(s, owned_s);
        spdlog::info("syncing s took {}", sw.elapsed());
    }
    spdlog::info("100 iterations done in {}", global_sw.elapsed());

    return 0;
}
int main(int argc, char* argv[]) try {
    dim::mpi::ctx ctx{argc, argv};
    auto app = structopt::app(brr::app_info.full_name, brr::app_info.version);
    spdlog::set_default_logger(
      spdlog::stdout_logger_mt(fmt::format("{} (n{:02})", brr::app_info.full_name, dim::mpi::rank())));

    return main_impl(app.parse<arguments_t>(argc, argv));
} catch (const structopt::exception& e) {
    spdlog::critical("{}", e.what());
    fmt::print(stderr, "{}", e.help());
    return 1;
} catch (const std::exception& e) {
    spdlog::critical("{}", e.what());
    return 2;
}