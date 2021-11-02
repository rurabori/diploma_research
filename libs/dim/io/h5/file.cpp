#include <dim/io/h5/file.h>

namespace dim::io::h5 {
auto file_t::create(const std::filesystem::path& path, uint32_t flags, plist_view_t create_plist,
                    plist_view_t access_plist) -> file_t {
    return file_t{::H5Fcreate(path.c_str(), flags, create_plist.get_id(), access_plist.get_id())};
}

auto file_t::open(const std::filesystem::path& path, uint32_t flags, plist_view_t access_plist) -> file_t {
    return file_t{::H5Fopen(path.c_str(), flags, access_plist.get_id())};
}
} // namespace dim::io::h5
