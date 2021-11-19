#ifndef INCLUDE_DIM_IO_MATRIX_MARKET
#define INCLUDE_DIM_IO_MATRIX_MARKET

#include <dim/io/file.h>
#include <dim/io/mmapped.h>
#include <dim/mat/storage_formats.h>

#include <mmio/mmio.h>
#include <scn/scn.h>

#include <filesystem>
#include <stdexcept>
#include <system_error>

namespace dim::io::matrix_market {

struct matrix_size_t
{
    mat::dimensions_t dimensions{};
    size_t num_non_zero{};

    static auto from_file(FILE* file) -> matrix_size_t;
};

template<typename ValueType>
auto read_coo(FILE* file, dim::mat::dimensions_t dimensions, size_t non_zero, bool symmetric, bool pattern) {
    auto retval = dim::mat::coo<ValueType>{dimensions, non_zero, symmetric};

    auto map = mmapped::from_file(file);
    // skip the already read part.
    auto span = map.as<char>().subspan(static_cast<size_t>(::ftell(file)));

    if (pattern)
        std::fill(retval.values.begin(), retval.values.end(), 1.);

    size_t idx = 0;
    for (auto remaining = std::string_view{span.data(), span.size()}; idx < non_zero && !remaining.empty(); ++idx) {
        const auto result = pattern ? scn::scan(remaining, "{} {} ", retval.row_indices[idx], retval.col_indices[idx])
                                    : scn::scan(remaining, "{} {} {} ", retval.row_indices[idx],
                                                retval.col_indices[idx], retval.values[idx]);
        if (!result)
            throw std::runtime_error{"scan failed"};

        // make 0 based.
        --retval.row_indices[idx];
        --retval.col_indices[idx];

        remaining = result.string_view();
    }

    return retval;
}

template<typename ValueType, template<typename> typename StorageType = mat::cache_aligned_vector>
auto load_as_csr(FILE* file) -> mat::csr<ValueType, StorageType> {
    MM_typecode banner{};
    mm_read_banner(file, &banner);

    if (!mm_is_coordinate(banner))
        throw std::invalid_argument{"Only coordinate matrix loading is implemented."};

    const auto [dimensions, num_non_zero] = matrix_size_t::from_file(file);

    const auto coo = read_coo<ValueType>(file, dimensions, num_non_zero,
                                         mm_is_symmetric(banner) || mm_is_hermitian(banner), mm_is_pattern(banner));

    return dim::mat::csr<ValueType, StorageType>::from_coo(coo);
}

template<typename ValueType, template<typename> typename StorageType = mat::cache_aligned_vector>
decltype(auto) load_as_csr(const std::filesystem::path& path) {
    return load_as_csr<ValueType, StorageType>(io::open(path, "r").get());
}

} // namespace dim::io::matrix_market

#endif /* INCLUDE_DIM_IO_MATRIX_MARKET */
