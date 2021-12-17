#ifndef INCLUDE_DIM_IO_FORMAT
#define INCLUDE_DIM_IO_FORMAT

#include <array>
#include <fmt/format.h>

namespace dim::io {

template<typename StorageType = size_t>
struct formattable_bytes
{ StorageType count; };

template<typename StorageType>
formattable_bytes(StorageType) -> formattable_bytes<StorageType>;

} // namespace dim::io

template<typename StorageType>
struct fmt::formatter<dim::io::formattable_bytes<StorageType>> : fmt::formatter<double>
{
    using formattable_bytes = dim::io::formattable_bytes<StorageType>;
    static constexpr auto possible_suffixes = std::array<std::string_view, 5>{"B", "kiB", "MiB", "GiB", "TiB"};

    struct normalized
    {
        double value{};
        std::string_view suffix;

        static auto normalize(double bytes) -> normalized {
            for (const auto& suffix : possible_suffixes) {
                if (bytes < 1024)
                    return normalized{.value = bytes, .suffix = suffix};

                bytes /= 1024;
            }

            return normalized{.value = bytes, .suffix = possible_suffixes.back()};
        }
    };

    template<typename FormatContext>
    auto format(const formattable_bytes& bytes, FormatContext& ctx) -> decltype(ctx.out()) {
        const auto [count, suffix] = normalized::normalize(static_cast<double>(bytes.count));

        auto out = fmt::formatter<double>::format(count, ctx);
        return format_to(out, "{}", suffix);
    }
};

#endif /* INCLUDE_DIM_IO_FORMAT */
