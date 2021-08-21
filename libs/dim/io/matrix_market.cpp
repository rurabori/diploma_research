#include <dim/io/matrix_market.h>

namespace dim::io::matrix_market {
auto matrix_size_t::from_file(FILE* file) -> matrix_size_t {
    int rows{};
    int cols{};
    int non_zero{};
    mm_read_mtx_crd_size(file, &rows, &cols, &non_zero);

    return {{static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)}, static_cast<size_t>(non_zero)};
}

} // namespace dim::io::matrix_market