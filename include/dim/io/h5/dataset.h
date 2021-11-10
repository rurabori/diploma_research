#ifndef INCLUDE_DIM_IO_H5_DATASET
#define INCLUDE_DIM_IO_H5_DATASET

#include <dim/io/h5/dataspace.h>
#include <dim/io/h5/location.h>
#include <dim/io/h5/plist.h>
#include <dim/io/h5/type.h>

#include <H5Dpublic.h>

#include <concepts>

namespace dim::io::h5 {

struct dataset_view_t : public view_t
{
    auto write(const void* data, type_view_t type, dataspace_view_t mem_space = dataspace_view_t::all(),
               dataspace_view_t file_space = dataspace_view_t::all(), plist_view_t props = plist_t::defaulted())
      -> void;

    auto write(std::ranges::contiguous_range auto&& data, dataspace_view_t file_space = dataspace_view_t::all(),
               plist_view_t props = plist_t::defaulted()) -> void {
        using translator_t = type_translator_t<std::ranges::range_value_t<decltype(data)>>;
        write(std::data(data), translator_t::in_memory(), dataspace_t::create(std::size(data)), file_space, props);
    }

    auto read(void* data, type_view_t type, dataspace_view_t mem_space = dataspace_view_t::all(),
              dataspace_view_t file_space = dataspace_view_t::all(), plist_view_t props = plist_t::defaulted()) const
      -> void;

    template<typename MemType>
    auto read(std::span<MemType> out, type_view_t type, dataspace_view_t file_space = dataspace_view_t::all(),
              plist_view_t props = plist_t::defaulted()) const -> void {
        read(out.data(), type, dataspace_t::create(hsize_t{out.size()}), file_space, props);
    }

    template<typename ContainerType, typename MemType = std::ranges::range_value_t<ContainerType>>
    auto read_slab(hsize_t start, type_view_t type, size_t count, plist_view_t props = plist_t::defaulted()) const
      -> ContainerType {
        auto result = ContainerType(count);
        auto space = get_dataspace();
        space.select_hyperslab(start, count);

        read(std::span{result}, type, space, props);
        return result;
    }

    template<typename ContainerType, typename MemType = std::ranges::range_value_t<ContainerType>,
             h5::type_translator Translator = type_translator_t<MemType>>
    auto read_slab(hsize_t start, size_t count, plist_view_t props = plist_t::defaulted()) const -> ContainerType {
        return read_slab<ContainerType>(start, Translator::in_memory(), count, props);
    }

    [[nodiscard]] auto get_dataspace() const -> dataspace_t;

    [[nodiscard]] auto get_type() const noexcept -> type_t;
};

class dataset_t : public view_wrapper_t<dataset_view_t, H5Dclose>
{
    using super_t = view_wrapper_t<dataset_view_t, H5Dclose>;
    using super_t::super_t;

    friend location_view_t;

public:
};

} // namespace dim::io::h5

#endif /* INCLUDE_DIM_IO_H5_DATASET */
