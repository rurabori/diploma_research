#include <dim/io/h5.h>
#include <dim/simple_main.h>

#include <memory>
#include <mkl.h>
#include <mkl_rci.h>
#include <mkl_spblas.h>

#include <fmt/format.h>
#include <magic_enum.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

struct mkl_err
{
    auto operator%(sparse_status_t stat) {
        if (stat != sparse_status_t::SPARSE_STATUS_SUCCESS)
            throw std::runtime_error{fmt::format("mkl error: {}", magic_enum::enum_name(stat))};
    }
};

// NOLINTNEXTLINE
#define mkl_try mkl_err{} %

struct sparse_deleter
{
    auto operator()(sparse_matrix_t mat) {
        if (mat)
            ::mkl_sparse_destroy(mat);
    }
};
using mkl_sparse = std::unique_ptr<std::remove_pointer_t<sparse_matrix_t>, sparse_deleter>;

auto create_mkl_csr(dim::mat::csr<double>& csr) -> mkl_sparse {
    auto* result = sparse_matrix_t{};
    mkl_try ::mkl_sparse_d_create_csr(
      &result, SPARSE_INDEX_BASE_ZERO, static_cast<int>(csr.dimensions.rows), static_cast<int>(csr.dimensions.cols),
      reinterpret_cast<int*>(csr.row_start_offsets.data()), reinterpret_cast<int*>(csr.row_start_offsets.data() + 1),
      reinterpret_cast<int*>(csr.col_indices.data()), csr.values.data());

    return mkl_sparse{result};
}

auto spmv(::sparse_matrix_t mat, std::span<const double> x, std::span<double> y) {
    auto descr
      = matrix_descr{.type = SPARSE_MATRIX_TYPE_GENERAL, .mode = SPARSE_FILL_MODE_FULL, .diag = SPARSE_DIAG_NON_UNIT};
    ::mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, mat, descr, x.data(), 0.0, y.data());
}

auto add_vectors(std::span<double> lhs, std::span<double> rhs) {
    const auto count = static_cast<int>(lhs.size());

    double eone = -1.E0;
    int ione = 1;

    ::daxpy(&count, &eone, lhs.data(), &ione, rhs.data(), &ione);
}

auto euclidean_norm(std::span<const double> vec) {
    const auto count = static_cast<int>(vec.size());
    int ione = 1;

    return ::dnrm2(&count, vec.data(), &ione);
}

class wrapper
{
    int _element_count{};
    std::vector<double> _temp{};
    std::vector<double> _solution{};
    std::span<double> _rhs{};

    int _ipar[128]{};
    double _dpar[128]{};

    template<const auto& Fun, typename... Args>
    auto impl(Args&&... args) -> int {
        auto rci_request = int{};

        Fun(&_element_count, std::data(_solution), std::data(_rhs), &rci_request, std::data(_ipar), std::data(_dpar),
            std::data(_temp), std::forward<Args>(args)...);

        return rci_request;
    }

public:
    explicit wrapper(size_t element_count, std::span<double> rhs)
      : _element_count{static_cast<int>(element_count)},
        _temp(element_count * 4, 0.0),
        _solution(element_count, 0.0),
        _rhs{rhs} {}

    auto init() -> int { return impl<::dcg_init>(); }
    auto check() -> int { return impl<::dcg_check>(); }
    auto dcg() -> int { return impl<::dcg>(); }

    auto compute_temp(::sparse_matrix_t mat) {
        auto temp_view = std::span{_temp};
        const auto count = static_cast<size_t>(_element_count);

        spmv(mat, temp_view.first(count), temp_view.subspan(count));
    }

    auto compute_euclidean_norm(::sparse_matrix_t mat) {
        spmv(mat, _solution, _temp);
        add_vectors(_rhs, _temp);
        return euclidean_norm(std::span{_temp}.first(static_cast<size_t>(_element_count)));
    }

    auto check_euclidean_norm(::sparse_matrix_t mat) { return compute_euclidean_norm(mat) <= 1.e-8; }

    auto set_iter_count(int count) { _ipar[4] = count; }

    auto try_compute(::sparse_matrix_t mat, int iter_count) -> bool {
        set_iter_count(iter_count);
        while (true) {
            check();
            switch (dcg()) {
                case 0:
                    return check_euclidean_norm(mat);
                case 1:
                    compute_temp(mat);
                    continue;
                case 2:
                    if (check_euclidean_norm(mat))
                        return true;

                    continue;
                default:
                    return false;
            }
        }
    }

    struct result_t
    {
        std::span<const double> x;
        int iteration_count;
    };

    auto get_result() -> result_t {
        auto iteration_count = 0;
        impl<::dcg_get>(&iteration_count);

        return {.x = _solution, .iteration_count = iteration_count};
    }
};

struct arguments_t
{
    std::filesystem::path input_file;
    std::optional<std::string> group_name{"A"};
    std::optional<double> threshold{0.1};
    std::optional<size_t> max_iters{100};
};
STRUCTOPT(arguments_t, input_file);

// NOTE: this does not work correctly, even example from MKL segfaults on its 8x8 matrix.
auto main_impl(const arguments_t& args) -> int {
    auto csr = dim::io::h5::read_matlab_compatible(args.input_file, *args.group_name);
    spdlog::info("matrix has size {}x{} (ia: {}, ja:{}, a:{})", csr.dimensions.rows, csr.dimensions.cols,
                 csr.row_start_offsets.size(), csr.col_indices.size(), csr.values.size());

    auto A = create_mkl_csr(csr);

    auto expected = std::vector<double>(csr.dimensions.cols, 1.0);
    auto rhs = std::vector<double>(csr.dimensions.rows, 0.0);

    spmv(A.get(), expected, rhs);

    auto ctx = wrapper{csr.dimensions.rows, rhs};

    ctx.init();
    if (!ctx.try_compute(A.get(), static_cast<int>(*args.max_iters)))
        throw std::runtime_error{"result not found."};

    const auto result = ctx.get_result();
    spdlog::info("result found after {} iterations", result.iteration_count);

    return 0;
}
DIM_MAIN(arguments_t);
