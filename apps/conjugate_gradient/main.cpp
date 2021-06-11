#include <memory>
#include <cstdio>
#include <vector>

#include <mmio/mmio.h>

#include <anonymouslib_avx2.h>

struct file_deleter_t
{
    void operator()(FILE* file) { fclose(file); }
};
using file_t = std::unique_ptr<FILE, file_deleter_t>;

int main(int argc, const char* argv[]) {
    // TODO: better commandline.
    if (argc < 2) return 1;

    file_t file{std::fopen(argv[1], "r")};

    MM_typecode typecode{};
    mm_read_banner(file.get(), &typecode);

    int rows{};
    int cols{};
    int non_zero{};
    mm_read_mtx_crd_size(file.get(), &rows, &cols, &non_zero);

    std::vector<double> values(non_zero, 0);
    std::vector<size_t> rid(non_zero, 0);
    std::vector<size_t> cid(non_zero, 0);

    for (size_t i = 0; i < non_zero; ++i) {
        std::fscanf(file.get(), "%llu %llu %lg", &cid[i], &rid[i], &values[i]);
    }

    return 0;
}
