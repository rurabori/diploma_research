#include "compare_results.h"

#include <dim/io/h5.h>

#include <spdlog/spdlog.h>

namespace h5 = dim::io::h5;

auto compare_results(const dim_cli::compare_results_t& args) -> int {
    constexpr auto nearly_equal = [](auto l, auto r) { return std::abs(l - r) <= (0.01 * std::abs(r)); };
    constexpr auto load_vec_and_info = [](auto&& path, auto&& group_name, auto&& dataset_name) {
        auto in = h5::file_t::open(path, H5F_ACC_RDONLY);
        auto group = in.open_group(group_name);

        return std::pair{fmt::format("{}:{}{}", path.filename().native(), group_name, dataset_name),
                         group.open_dataset(dataset_name).template read<std::vector<double>>()};
    };

    const auto [lhs_id, lhs] = load_vec_and_info(args.input_file, *args.lhs_group, *args.lhs_dataset);
    const auto [rhs_id, rhs]
      = load_vec_and_info(args.input_file_2 ? *args.input_file_2 : args.input_file, *args.rhs_group, *args.rhs_dataset);

    bool correct = true;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!nearly_equal(lhs[i], rhs[i])) {
            spdlog::info("at {}: expected {}, got {}", i, lhs[i], rhs[i]);
            correct = false;
        }
    }

    if (!correct) {
        spdlog::error("vectors {} and {} are not equal", lhs_id, rhs_id);
        return 1;
    }

    spdlog::info("vectors {} and {} are equal", lhs_id, rhs_id);

    return 0;
}