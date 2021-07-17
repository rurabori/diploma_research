#include <algorithm>
#include <bits/ranges_algo.h>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iterator>
#include <random>
#include <ranges>
#include <scn/detail/scan.h>
#include <span>
#include <stdexcept>
#include <vector>

#include <mm_malloc.h>
#include <mmio/mmio.h>

// TODO: make more standard/portable.
#include <sys/mman.h>
#include <sys/stat.h>

#include <fmt/chrono.h>
#include <fmt/format.h>

#include <anonymouslib_avx2.h>

#include <tclap/Arg.h>
#include <tclap/CmdLine.h>
#include <tclap/SwitchArg.h>
#include <tclap/ValueArg.h>

#include <magic_enum.hpp>

#include <scn/scn.h>
#include <stx/panic.h>

#include "matrix_storage_formats.h"
#include "spmv_algos.hpp"
#include "timed_section.h"

#include "version.h"

struct file_deleter_t
{
    void operator()(FILE* file) { fclose(file); }
};

using file_t = std::unique_ptr<FILE, file_deleter_t>;
using cg::matrix_storage_formats::cache_aligned_vector;

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

auto generate_random_vector(size_t required_size) {
    cache_aligned_vector<double> x(required_size, 0.);

    std::mt19937_64 prng{std::random_device{}()};
    std::uniform_real_distribution<double> dist{0., 1.};
    std::ranges::generate(x, [&] { return dist(prng); });

    return x;
}

auto read_matrix_size(FILE* file) {
    int rows{};
    int cols{};
    int non_zero{};
    mm_read_mtx_crd_size(file, &rows, &cols, &non_zero);

    return std::pair{cg::dimensions_t{.rows = static_cast<uint32_t>(rows), .cols = static_cast<uint32_t>(cols)},
                     static_cast<size_t>(non_zero)};
}

class mmapped
{
    void* _memory{};
    size_t _size{};

public:
    mmapped(int fd, size_t size, int prototype = PROT_READ, int flags = MAP_SHARED | MAP_FILE | MAP_POPULATE,
            std::ptrdiff_t offset = 0)
      : _memory{::mmap(nullptr, size, prototype, flags, fd, offset)}, _size{size} {
        if (!_memory)
            stx::panic(fmt::format("mmap failed with {}", errno));
    }

    static mmapped from_file(FILE* file) {
        int fd = fileno(file);
        struct stat stat = {};
        ::fstat(fd, &stat);

        return mmapped{fd, static_cast<size_t>(stat.st_size)};
    }

    mmapped(const mmapped&) = delete;
    mmapped(mmapped&& o) noexcept : _memory{std::exchange(o._memory, nullptr)}, _size{std::exchange(o._size, 0)} {}
    mmapped& operator=(const mmapped&) = delete;
    mmapped& operator=(mmapped&& o) noexcept {
        if (this == &o)
            return *this;

        std::swap(_memory, o._memory);
        std::swap(_size, o._size);
        return *this;
    }

    ~mmapped() { release(); }

    void release() {
        if (_memory)
            ::munmap(std::exchange(_memory, nullptr), std::exchange(_size, 0));
    }

    template<typename As>
    auto as() -> std::span<As> {
        return std::span{reinterpret_cast<As*>(_memory), _size / sizeof(As)};
    }
};

template<typename ValueType>
auto read_coo(FILE* file, cg::dimensions_t dimensions, size_t non_zero, bool symmetric) {
    auto retval = cg::matrix_storage_formats::coo<ValueType>{dimensions, non_zero, symmetric};

    auto map = mmapped::from_file(file);
    // skip the already read part.
    auto span = map.as<char>().subspan(static_cast<size_t>(::ftell(file)));

    size_t idx = 0;
    for (auto remaining = std::string_view{span.data(), span.size()}; !remaining.empty(); ++idx) {
        const auto result
          = scn::scan(remaining, "{} {} {} ", retval.row_indices[idx], retval.col_indices[idx], retval.values[idx]);
        if (!result)
            stx::panic("scan failed");

        remaining = result.string_view();
    }

    return retval;
}

auto load_matrix(const std::filesystem::path& path) {
    file_t file{std::fopen(path.c_str(), "r")};
    if (!file)
        throw std::runtime_error{"File couldn't be opened."};

    MM_typecode typecode{};
    mm_read_banner(file.get(), &typecode);

    auto [dimensions, non_zero] = read_matrix_size(file.get());
    print_mm_info(typecode, dimensions, non_zero);

    if (!mm_is_coordinate(typecode))
        throw std::runtime_error{"Only coordinate matrix loading is implemented."};

    const auto start = std::chrono::steady_clock::now();
    const auto coo = read_coo<double>(file.get(), dimensions, static_cast<size_t>(non_zero),
                                      mm_is_symmetric(typecode) || mm_is_hermitian(typecode));

    const auto end = std::chrono::steady_clock::now() - start;
    fmt::print(stderr, "Loading from file took {}\n", end);

    return cg::matrix_storage_formats::csr<double>::from_coo(coo);
}

struct arguments
{
    enum class algorithm_t
    {
        cpu_sequential,
        cpu_avx2,
#ifdef CUDA_ENABLED
        cuda,
#endif
    };

    std::filesystem::path matrix_file{};
    bool debug{};
    algorithm_t algorithm;

    static algorithm_t get_algoritm(std::string_view string_representation) {
        if (auto algo = magic_enum::enum_cast<algorithm_t>(string_representation); algo)
            return *algo;

        return algorithm_t::cpu_sequential;
    }

    static arguments from_main(int argc, const char* argv[]) {
        TCLAP::CmdLine commandline{"Conjugate Gradient.", ' ', conjugate_gradient_VER};

        TCLAP::ValueArg<std::string> matrix_name_arg{"m", "matrix-path", "Path to matrix", true, "", "string"};
        commandline.add(matrix_name_arg);

        TCLAP::SwitchArg debug_arg{"d", "debug", "Enable debug output", false};
        commandline.add(debug_arg);

        TCLAP::ValueArg<std::string> algorithm_arg{"a",   "algorithm",      "Algorithm to use for SpMV",
                                                   false, "cpu_sequential", "one of [cpu_sequential, cpu_avx2, cuda]"};
        commandline.add(algorithm_arg);

        commandline.parse(argc, argv);
        return arguments{.matrix_file = matrix_name_arg.getValue(),
                         .debug = debug_arg.getValue(),
                         .algorithm = get_algoritm(algorithm_arg.getValue())};
    }
};

int main(int argc, const char* argv[]) {
    auto arguments = arguments::from_main(argc, argv);
    auto matrix = load_matrix(arguments.matrix_file);

    auto consumed_memory
      = matrix.col_indices.size() * sizeof(typename decltype(matrix.col_indices)::value_type)
        + matrix.row_start_offsets.size() * sizeof(typename decltype(matrix.row_start_offsets)::value_type)
        + matrix.values.size() * sizeof(typename decltype(matrix.values)::value_type);

    fmt::print("consumed_memory={}\n", consumed_memory);

    auto dimensions = matrix.dimensions;

    auto x = generate_random_vector(dimensions.cols);

    auto Y = cache_aligned_vector<double>(matrix.dimensions.rows, 0.);
    switch (arguments.algorithm) {
        case arguments::algorithm_t::cpu_sequential: {
            report_timed_section("SpMV", [&] { cg::spmv_algos::cpu_sequential(matrix, std::span{x}, std::span{Y}); });
            break;
        }
        case arguments::algorithm_t::cpu_avx2: {
            auto handle = cg::spmv_algos::create_csr5_handle(matrix);
            report_timed_section("SpMV", [&] { cg::spmv_algos::cpu_avx2(handle, std::span{x}, std::span{Y}); });
            handle.destroy();
            break;
        }
#ifdef CUDA_ENABLED
        case arguments::algorithm_t::cuda: {
            cg::spmv_algos::cuda_complete_bench(matrix, std::span{x}, std::span{Y});
            break;
        }
#endif
    }

    if (arguments.debug) {
        auto Y_ref = cache_aligned_vector<double>(matrix.dimensions.rows, 0.);
        cg::spmv_algos::cpu_sequential(matrix, std::span{x}, std::span{Y_ref});

        const auto correct
          = std::ranges::equal(Y, Y_ref, [](auto l, auto r) { return std::abs(l - r) <= (0.01 * std::abs(r)); });
        fmt::print(stderr, "SpMV correct: {}\n", correct);
    }

    return 0;
}
