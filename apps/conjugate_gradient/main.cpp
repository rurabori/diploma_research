#include <memory>
#include <exception>
#include <ranges>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <numeric>
#include <stdexcept>
#include <tclap/Arg.h>
#include <tclap/SwitchArg.h>
#include <tclap/ValueArg.h>
#include <vector>
#include <filesystem>
#include <span>
#include <random>
#include <concepts>
#include <thread>

#include <mmio/mmio.h>
#include <mm_malloc.h>

#include <fmt/format.h>

#include <anonymouslib_avx2.h>
#include <tclap/CmdLine.h>

#include "cache_aligned_allocator.h"
#include "matrix_storage_formats.h"

#include "version.h"

struct file_deleter_t
{
    void operator()(FILE* file) { fclose(file); }
};

using file_t = std::unique_ptr<FILE, file_deleter_t>;

void print_mm_info(const MM_typecode& typecode, cg::dimensions_t dimensions, size_t non_zero) {
    fmt::print("Matrix info:\n"
               "  dimensions: {}x{}\n"
               "num elements: {}\n"
               "     complex: {}\n"
               "     pattern: {}\n"
               "        real: {}\n"
               "     integer: {}\n"
               "   symmetric: {}\n"
               "   hermitian: {}\n",
               dimensions.rows, dimensions.cols, non_zero, mm_is_complex(typecode), mm_is_pattern(typecode),
               mm_is_real(typecode), mm_is_integer(typecode), mm_is_symmetric(typecode), mm_is_hermitian(typecode));
}

template<typename Ty>
using cache_aligned_vector = std::vector<Ty, cg::cache_aligned_allocator_t<Ty>>;

auto generate_random_vector(size_t required_size) {
    cache_aligned_vector<double> x(required_size, 0.);

    std::mt19937_64 prng{std::random_device{}()};
    std::uniform_int_distribution<uint64_t> dist{};
    std::ranges::generate(x, [&] { return dist(prng) % 10; });

    return x;
}

template<typename ValueType = double>
struct csr_matrix
{
    // TODO: better structure layout.
    cg::dimensions_t dimensions{};
    size_t num_non_zero{};
    cache_aligned_vector<ValueType> values;
    cache_aligned_vector<int> rid;
    cache_aligned_vector<int> cid;
    anonymouslibHandle<int, unsigned int, ValueType> A;

    void inputCSR() { A.inputCSR(static_cast<int>(num_non_zero), rid.data(), cid.data(), values.data()); }

    static auto read_coo(FILE* file, cg::dimensions_t dimensions, size_t non_zero) {
        auto retval = cg::matrix_storage_formats::coo<ValueType>::create(dimensions, non_zero);

        for (size_t i = 0; i < non_zero; ++i) {
            std::fscanf(file, "%d %d %lg", &retval.row_indices[i], &retval.col_indices[i], &retval.values[i]);
            // convert to 0 based.
            --retval.row_indices[i];
            --retval.col_indices[i];
        }

        return retval;
    }

    static auto read_matrix_size(FILE* file) {
        int rows{};
        int cols{};
        int non_zero{};
        mm_read_mtx_crd_size(file, &rows, &cols, &non_zero);

        return std::pair{cg::dimensions_t{.rows = static_cast<uint32_t>(rows), .cols = static_cast<uint32_t>(cols)},
                         static_cast<size_t>(non_zero)};
    }

    static csr_matrix from_matrix_market(const std::filesystem::path& path) {
        file_t file{std::fopen(path.c_str(), "r")};
        if (!file) throw std::runtime_error{"File couldn't be opened."};

        MM_typecode typecode{};
        mm_read_banner(file.get(), &typecode);
        if (!mm_is_coordinate(typecode)) throw std::runtime_error{"Only coordinate matrix loading is implemented."};

        auto [dimensions, non_zero] = read_matrix_size(file.get());
        print_mm_info(typecode, dimensions, non_zero);

        const auto coo = read_coo(file.get(), dimensions, static_cast<size_t>(non_zero));

        // count occurences of rows.
        std::vector<int> csr_row_counter(dimensions.rows + 1, 0);
        for (auto row_id : coo.row_indices) ++csr_row_counter[row_id];

        // prefix scan to get the starting indices of cols in a row.
        auto rid = cache_aligned_vector<int>(dimensions.rows + 1, 0);
        std::exclusive_scan(csr_row_counter.begin(), csr_row_counter.end(), rid.begin(), 0);

        // reuse the csr_row_counter as offset counter for rows.
        std::ranges::fill(csr_row_counter, 0);

        // transform to CSR.
        auto values = cache_aligned_vector<double>(non_zero, 0);
        auto cid = cache_aligned_vector<int>(non_zero, 0);
        for (size_t i = 0; i < non_zero; ++i) {
            const auto row = coo.row_indices[i];
            const auto row_start = rid[row];
            const auto offset = row_start + csr_row_counter[row]++;

            cid[offset] = coo.col_indices[i];
            values[offset] = coo.values[i];
        }

        return csr_matrix{.dimensions = dimensions,
                          .num_non_zero = non_zero,
                          .values = std::move(values),
                          .rid = std::move(rid),
                          .cid = std::move(cid),
                          .A = {static_cast<int>(dimensions.rows), static_cast<int>(dimensions.cols)}};
    }
};

template<typename Callable>
auto timed_section(Callable&& callable) {
    auto begin = std::chrono::steady_clock::now();
    std::forward<Callable>(callable)();
    return std::chrono::steady_clock::now() - begin;
}

template<typename Callable>
auto report_timed_section(std::string_view name, Callable&& callable) {
    using std::chrono::duration_cast;
    using std::chrono::nanoseconds;

    auto duration = timed_section(std::forward<Callable>(callable));
    fmt::print(FMT_STRING("{} took: {}ns\n"), name, duration_cast<nanoseconds>(duration).count());
}

struct arguments
{
    std::filesystem::path matrix_file{};
    bool debug{};

    static arguments from_main(int argc, const char* argv[]) {
        TCLAP::CmdLine commandline{"Conjugate Gradient.", ' ', conjugate_gradient_VER};

        TCLAP::ValueArg<std::string> matrix_name_arg{"m", "matrix-path", "Path to matrix", true, "", "string"};
        commandline.add(matrix_name_arg);

        TCLAP::SwitchArg debug_arg{"d", "debug", "Enable debug output", false};
        commandline.add(debug_arg);

        commandline.parse(argc, argv);

        return arguments{.matrix_file = matrix_name_arg.getValue(), .debug = debug_arg.getValue()};
    }
};

// TODO: a better commandline parser, fire-hpp seems to be broken on newer clang.
int main(int argc, const char* argv[]) {
    auto arguments = arguments::from_main(argc, argv);

    auto matrix = csr_matrix<double>::from_matrix_market(arguments.matrix_file);
    auto x = generate_random_vector(matrix.dimensions.cols);

    // do sequential algo for reference.
    auto y_ref = std::vector<double>(matrix.dimensions.rows, 0.);
    report_timed_section("Sequential SpMV", [&] {
        for (size_t i = 0; i < matrix.dimensions.rows; ++i) {
            const auto row_start = matrix.rid[i];
            const auto row_end = matrix.rid[i + 1];
            double sum{};
            for (auto j = row_start; j < row_end; ++j) {
                const auto column = matrix.cid[j];
                const auto value = matrix.values[j];
                sum += x[column] * value * 1.0;
            }

            y_ref[i] = sum;
        }
    });
    // careful, this shuffles x around making it unusable for calculations.
    matrix.inputCSR();
    matrix.A.setX(x.data());
    matrix.A.setSigma(ANONYMOUSLIB_CSR5_SIGMA);
    report_timed_section("CSR5 conversion", [&] {
        matrix.A.asCSR();
        matrix.A.asCSR5();
    });

    // make output buff.
    auto Y = cache_aligned_vector<double>(matrix.dimensions.rows, 0.);
    report_timed_section("CSR5 SpMV", [&] { matrix.A.spmv(1.0, Y.data()); });

    constexpr auto comparator = [](auto lhs, auto rhs) { return !(std::abs(lhs - rhs) > 0.01 * std::abs(lhs)); };

    auto are_same = std::ranges::equal(Y, y_ref, comparator);

    fmt::print("SPMV correct : {}\n", are_same);
    if (!are_same && arguments.debug) {
        for (size_t i = 0; i < matrix.dimensions.rows; ++i) {
            auto val = Y[i];
            auto ref = y_ref[i];
            if (!comparator(val, ref)) fmt::print("{}: {} {}\n", i, val, ref);
        }
    }

    matrix.A.destroy();
}
