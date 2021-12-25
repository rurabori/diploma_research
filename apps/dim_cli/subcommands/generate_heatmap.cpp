#include "generate_heatmap.h"
#include "dim/io/h5.h"

#include <fmt/chrono.h>
#include <fmt/format.h>

#include <opencv2/core/matx.hpp>
#include <opencv2/opencv.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <stdexcept>

namespace {
auto output_path(const std::filesystem::path& input_path) {
    auto result = input_path.stem();
    result.replace_extension(".png");
    return result;
}

auto calculate_step(size_t num_parts, size_t total) -> size_t {
    return std::ceil(static_cast<double>(total) / static_cast<double>(num_parts));
}

auto partition(size_t idx, size_t step) -> size_t {
    return std::floor(static_cast<double>(idx) / static_cast<double>(step));
}

auto get_color(size_t i) {
    using cv::Scalar;

    static const Scalar colors[] = {
      {227, 206, 166}, {180, 120, 31}, {138, 223, 178}, {44, 160, 51},  {153, 154, 251}, {28, 26, 227},
      {111, 191, 253}, {0, 127, 255},  {214, 178, 202}, {154, 61, 106}, {153, 255, 255}, {40, 89, 177},
    };

    return colors[i];
}

auto colored_map(const dim_cli::generate_heatmap_t& args, const auto& matrix) {
    const auto resolution = *args.resolution;

    const auto col_step = calculate_step(resolution[0], matrix.dimensions.cols);
    const auto val_step = calculate_step(*args.process_count, matrix.non_zero_count());
    const auto row_step = calculate_step(resolution[1], matrix.dimensions.rows);
    spdlog::info("heatmap pixel equals {}x{} chunk of matrix", col_step, row_step);

    auto res = cv::Mat(cv::Size(resolution[0], resolution[1]), CV_8UC3, cv::Scalar{0, 0, 0});
    auto i = size_t{};
    matrix.iterate([&](const auto& coords, double /*value*/) {
        auto& elem = res.at<cv::Vec<uint8_t, 3>>(partition(coords.row, row_step), partition(coords.col, col_step));
        const auto process = partition(i++, val_step);
        const auto col = get_color(process);

        // do not recolor.
        if (elem[0])
            return;

        elem[0] = col[0];
        elem[1] = col[1];
        elem[2] = col[2];
    });

    return res;
}

auto bw_map(const dim_cli::generate_heatmap_t& args, const auto& matrix) {
    const auto resolution = *args.resolution;

    const auto col_step = calculate_step(resolution[0], matrix.dimensions.cols);
    const auto row_step = calculate_step(resolution[1], matrix.dimensions.rows);
    spdlog::info("heatmap pixel equals {}x{} chunk of matrix", col_step, row_step);

    auto res = cv::Mat(cv::Size(resolution[0], resolution[1]), CV_32SC1);
    matrix.iterate([&](const auto& coords, double /*value*/) {
        auto& elem = res.at<uint32_t>(partition(coords.row, row_step), partition(coords.col, col_step));
        elem += 1;
    });

    return res;
}

} // namespace

auto generate_heatmap(const dim_cli::generate_heatmap_t& args) -> int {
    if (!exists(args.input_file))
        throw std::invalid_argument{fmt::format("{} does not exist.", args.input_file.string())};

    spdlog::info("loading {}", args.input_file.string());
    auto sw = spdlog::stopwatch{};

    const auto matrix = dim::io::h5::load_csr5(args.input_file, *args.group);

    spdlog::info("loaded {} in {} ({} x {}, {} non zero elements)", args.input_file.string(), sw.elapsed(),
                 matrix.dimensions.rows, matrix.dimensions.rows, matrix.non_zero_count());

    const auto res = args.process_count ? colored_map(args, matrix) : bw_map(args, matrix);
    const auto path = args.output_file ? *args.output_file : output_path(args.input_file);
    cv::imwrite(path.string(), res);

    return 0;
}
