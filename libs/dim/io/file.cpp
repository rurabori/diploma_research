#include <dim/io/file.h>

namespace dim::io {

void file_deleter_t::operator()(FILE* file) {
    // NOLINTNEXTLINE (cppcoreguidelines-owning-memory)
    std::fclose(file);
}

auto open(const std::filesystem::path& path, const char* mode) -> file_t {
    file_t retval{std::fopen(path.c_str(), mode)};
    if (!retval)
        throw std::system_error{errno, std::generic_category(), path.string()};

    return retval;
}

} // namespace dim::io
