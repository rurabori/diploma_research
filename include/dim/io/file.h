#ifndef INCLUDE_DIM_IO_FILE
#define INCLUDE_DIM_IO_FILE

#include <cstdio>
#include <filesystem>
#include <memory>

namespace dim::io {
struct file_deleter_t
{
    void operator()(FILE* file);
};

using file_t = std::unique_ptr<FILE, file_deleter_t>;

auto open(const std::filesystem::path& path, const char* mode) -> file_t;

} // namespace dim::io

#endif /* INCLUDE_DIM_IO_FILE */
