#include "csr5_info.h"

#include "dim/bit.h"
#include <dim/io/h5.h>

#include <fmt/color.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <limits>

namespace h5 = dim::io::h5;
using tile_t = dim::mat::csr5<double>::tile_descriptor_type;

namespace {
auto generate_tile_styles(const tile_t& current_tile) {
    std::array<std::array<fmt::text_style, tile_t::num_rows>, tile_t::num_cols> styles;

    current_tile.iterate_columns([&](auto idx, auto desc) {
        size_t last_set = std::numeric_limits<size_t>::max();
        for (size_t row = 0; row < tile_t::num_rows; ++row) {
            if (!dim::has_rbit_set(desc.bit_flag, row))
                continue;

            auto prev_set = std::exchange(last_set, row);

            // end of red section.
            if (prev_set == std::numeric_limits<size_t>::max()) {
                for (size_t i = 0; i < row; ++i)
                    styles[idx][i] = fmt::fg(fmt::color::red);
                continue;
            }

            // self contained, green section.
            for (size_t i = prev_set; i < row; ++i)
                styles[idx][i] = fmt::fg(fmt::color::green);
        }

        const auto is_red = last_set == std::numeric_limits<size_t>::max();
        const auto color = fmt::fg(is_red ? fmt::color::red : fmt::color::blue);
        for (size_t i = is_red ? 0 : last_set; i < tile_t::num_rows; ++i)
            styles[idx][i] = color;
    });

    return styles;
}
} // namespace

auto csr5_info(const dim_cli::csr5_info_t& arguments) -> int {
    auto file = h5::file_t::open(arguments.input, H5F_ACC_RDONLY);
    auto matrix_group = file.open_group(*arguments.matrix_group);

    auto dataset = matrix_group.open_dataset("tile_ptr");
    auto tile_ptr = std::vector<uint32_t>(dataset.get_dataspace().get_dim());
    dataset.read(tile_ptr.data(), H5T_NATIVE_UINT32);

    // upper bound always returns bigger, we don't care about bigger rows but we do want the one that's bigger than our
    // row.
    const auto tile_end_it = std::upper_bound(tile_ptr.begin(), tile_ptr.end(), arguments.row);
    const auto tile_end = std::distance(tile_ptr.begin(), tile_end_it);

    // TODO: the -1 will bonk on 0.
    const auto tile_start_it = std::lower_bound(tile_ptr.begin(), tile_end_it, arguments.row) - 1;
    const auto tile_start = std::distance(tile_ptr.begin(), tile_start_it);

    auto tile_dataset = matrix_group.open_dataset("tile_desc");
    auto space = tile_dataset.get_dataspace();

    const auto tile_count = static_cast<size_t>(tile_end - tile_start);
    space.select_hyperslab(static_cast<hsize_t>(tile_start), static_cast<hsize_t>(tile_count));

    std::vector<tile_t> tiles(tile_count, tile_t{});
    tile_dataset.read(tiles.data(), h5::type_t::create_array(H5T_NATIVE_UINT32, 4),
                      h5::dataspace_t::create(hsize_t{tile_count}), space);

    spdlog::info("searching for start in tiles {} - {}, spanning {} - {}", tile_start, tile_end, *tile_start_it,
                 *tile_end_it);

    auto vals_dataset = matrix_group.open_dataset("vals");
    auto vals_space = vals_dataset.get_dataspace();

    constexpr auto slab = tile_t::num_cols * tile_t::num_rows;
    const auto vals_count = tile_count * slab;
    vals_space.select_hyperslab(hsize_t{tile_start * slab}, hsize_t{vals_count});

    std::vector<double> vals(vals_count, double{});
    vals_dataset.read(vals.data(), H5T_NATIVE_DOUBLE, h5::dataspace_t::create(hsize_t{vals_count}), vals_space);

    for (auto cid = tile_start; const auto& current_tile : tiles) {
        constexpr auto table_style = fmt::emphasis::bold | fmt::fg(fmt::color::floral_white);

        auto styles = generate_tile_styles(current_tile);

        fmt::print(table_style, "{:8}| 0 1 2 3 |\n", cid);
        for (size_t row = 0; row < tile_t::num_rows; ++row) {
            fmt::print(table_style, "{:8}|", row);
            for (size_t idx = 0; idx < tile_t::num_cols; ++idx) {
                auto new_row_start = dim::has_rbit_set(current_tile.columns[idx].bit_flag, row);
                fmt::print(styles[idx][row], " {}", new_row_start ? 1 : 0);
            }
            fmt::print(table_style, " |\n");
        }

        fmt::print(table_style, "\n{:8}| values \n", tile_start);
        for (size_t row = 0; row < tile_t::num_rows; ++row) {
            fmt::print(table_style, "{:8}|", row);
            for (size_t idx = 0; idx < tile_t::num_cols; ++idx) {
                fmt::print(
                  styles[idx][row], " {:15.12e}",
                  vals[(cid - tile_start) * tile_t::num_cols * tile_t::num_rows + row * tile_t::num_cols + idx]);
            }
            fmt::print(table_style, " |\n");
        }

        ++cid;
    }

    return 0;
}
