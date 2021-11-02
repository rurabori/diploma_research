#ifndef INCLUDE_DIM_IO_H5_ATTRIBUTE
#define INCLUDE_DIM_IO_H5_ATTRIBUTE

#include "dim/io/h5/type.h"
#include <dim/io/h5/err.h>
#include <dim/io/h5/location.h>
#include <dim/io/h5/object.h>

#include <H5Apublic.h>
#include <concepts>

namespace dim::io::h5 {

struct attribute_view_t : public location_view_t
{
    auto write(const void* data, type_view_t type) -> void;
    auto read(void* data, type_view_t type) const -> void;

    template<typename Ty, type_translator Translator = type_translator_t<Ty>>
    auto read() const -> Ty {
        Ty result;
        read(&result, Translator::in_memory());
        return result;
    }

    [[nodiscard]] auto get_type() const noexcept -> type_t;
};

class attribute_t : public view_wrapper_t<attribute_view_t, H5Aclose>
{
    using super_t = view_wrapper_t<attribute_view_t, H5Aclose>;
    using super_t::super_t;

    friend object_view_t;

public:
};

} // namespace dim::io::h5

#endif /* INCLUDE_DIM_IO_H5_ATTRIBUTE */
