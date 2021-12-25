#include "generate_heatmap.h"
#include "dim/io/h5.h"

#include <fmt/format.h>

#include <opencv2/core/persistence.hpp>
#include <stdexcept>

#include <opencv2/opencv.hpp>

auto output_path(const std::filesystem::path& input_path) {
    auto result = input_path.stem();
    result.replace_extension(".png");
    return result;
}

auto generate_heatmap(const dim_cli::generate_heatmap_t& args) -> int {
    if (!exists(args.input_file))
        throw std::invalid_argument{fmt::format("{} does not exist.", args.input_file.string())};

    const auto matrix = dim::io::h5::load_csr5(args.input_file, *args.group);

    auto res
      = cv::Mat(cv::Size(static_cast<int>(matrix.dimensions.cols), static_cast<int>(matrix.dimensions.cols)), CV_8UC1);

    matrix.iterate([&res](const auto& coords, double /*value*/) {
        res.at<uint8_t>(static_cast<int>(coords.row), static_cast<int>(coords.col)) = 255;
    });

    cv::resize(res, res, cv::Size(512, 512));

    const auto path = args.output_file ? *args.output_file : output_path(args.input_file);
    cv::imwrite(path.string(), res);

    return 0;
}
