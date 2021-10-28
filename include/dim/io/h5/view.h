#ifndef INCLUDE_DIM_IO_H5_VIEW
#define INCLUDE_DIM_IO_H5_VIEW

#include <H5Ipublic.h>
#include <utility>

namespace dim::io::h5 {

class view_t
{
    hid_t _id;

protected:
    hid_t& take_id() && { return _id; }

public:
    // NOLINTNEXTLINE - should be convertible from HDF5 defaults.
    constexpr view_t(hid_t id) : _id{id} {}

    [[nodiscard]] constexpr hid_t get_id() const noexcept { return _id; }
};

template<typename View, const auto& Destroy>
struct view_wrapper_t : public View
{
    explicit view_wrapper_t(hid_t hid) : View{hid} {}
    view_wrapper_t(const view_wrapper_t&) = delete;
    view_wrapper_t(view_wrapper_t&& o) noexcept : View{std::exchange(std::move(o).take_id(), H5I_INVALID_HID)} {}
    view_wrapper_t& operator=(const view_wrapper_t&) = delete;
    view_wrapper_t& operator=(view_wrapper_t&& o) noexcept { std::swap(std::move(o).take_id(), o._id); }
    ~view_wrapper_t() noexcept {
        if (auto id = View::get_id(); ::H5Iis_valid(id))
            Destroy(View::get_id());
    }

    // NOLINTNEXTLINE
    operator View() { return *this; }
};

} // namespace dim::io::h5

#endif /* INCLUDE_DIM_IO_H5_VIEW */
