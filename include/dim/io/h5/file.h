#ifndef INCLUDE_DIM_IO_H5_FILE
#define INCLUDE_DIM_IO_H5_FILE

#include <dim/io/h5/location.h>
#include <dim/io/h5/plist.h>

#include <H5Fpublic.h>

#include <filesystem>

namespace dim::io::h5 {

class file_view_t : public location_view_t
{};

class file_t : public view_wrapper_t<file_view_t, H5Fclose>
{
    using super_t = view_wrapper_t<file_view_t, H5Fclose>;
    using super_t::super_t;

public:
    [[nodiscard]] static file_t create(const std::filesystem::path& path, uint32_t flags,
                                       plist_view_t create_plist = plist_t::defaulted(),
                                       plist_view_t access_plist = plist_t::defaulted());

    [[nodiscard]] static file_t open(const std::filesystem::path& path, uint32_t flags,
                                     plist_view_t access_plist = plist_t::defaulted());
};

} // namespace dim::io::h5

#endif /* INCLUDE_DIM_IO_H5_FILE */
