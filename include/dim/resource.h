#ifndef INCLUDE_DIM_RESOURCE
#define INCLUDE_DIM_RESOURCE

#include <concepts>
#include <optional>

namespace dim {

template<typename TraitsType, typename ResourceType>
concept resource_deleter = requires(ResourceType res) {
    { TraitsType::destroy(res) } -> std::same_as<void>;
    { TraitsType::is_valid(res) } -> std::same_as<bool>;
    { TraitsType::invalid_value() } -> std::same_as<ResourceType>;
};

template<typename HeldType, resource_deleter<HeldType> Traits>
class resource_t
{
    HeldType _resource{Traits::invalid_value()};

public:
    explicit resource_t(HeldType&& resource) : _resource{std::move(resource)} {}

    resource_t(resource_t&& other) noexcept = default;
    resource_t(const resource_t& other) noexcept = delete;

    resource_t& operator=(resource_t&& other) noexcept = default;
    resource_t& operator=(const resource_t& other) noexcept = delete;

    ~resource_t() noexcept(noexcept(reset())) { reset(); }

    void reset() noexcept(noexcept(Traits::destroy(_resource))) {
        if (!_resource)
            return;

        Traits::destroy(std::exchange(_resource, Traits::invalid_value()));
    }

    // NOLINTNEXTLINE - we want this to be convertible to bool.
    operator bool() const noexcept { return Traits::is_valid(_resource); }

    HeldType& get() const noexcept { return _resource; }
    HeldType& get() noexcept { return _resource; }
};

} // namespace dim

#endif /* INCLUDE_DIM_RESOURCE */
