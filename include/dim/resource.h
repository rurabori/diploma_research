#ifndef INCLUDE_DIM_RESOURCE
#define INCLUDE_DIM_RESOURCE

#include <concepts>
#include <optional>

namespace dim {

template<typename TraitsType, typename ResourceType>
concept resource_deleter = requires(ResourceType res) {
    { TraitsType::destroy(res) } -> std::same_as<void>;
};

template<typename HeldType, resource_deleter<HeldType> Traits>
class resource_t
{
    std::optional<HeldType> _resource;

public:
    explicit resource_t(HeldType&& resource) : _resource{std::move(resource)} {}

    resource_t(resource_t&& other) noexcept = default;
    resource_t(const resource_t& other) noexcept = delete;

    resource_t& operator=(resource_t&& other) noexcept = default;
    resource_t& operator=(const resource_t& other) noexcept = delete;

    ~resource_t() noexcept(noexcept(reset())) { reset(); }

    void reset() noexcept(noexcept(Traits::destroy(*_resource))) {
        if (!_resource)
            return;

        Traits::destroy(*_resource);
        _resource.reset();
    }

    // NOLINTNEXTLINE - we want this to be convertible to bool.
    operator bool() const noexcept { return _resource.has_value(); }

    HeldType& get() const noexcept { return *_resource; }
};

} // namespace dim

#endif /* INCLUDE_DIM_RESOURCE */
