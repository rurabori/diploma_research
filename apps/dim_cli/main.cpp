
#include <spdlog/spdlog.h>

#include <dim/io/h5.h>
#include <dim/io/matrix_market.h>

#include "arguments.h"
#include "subcommands.h"

#include "version.h"

auto compare_results(const dim_cli::compare_results_t& args) -> int {
    namespace h5 = dim::io::h5;
    using dim::io::h5::read_vector;
    constexpr auto nearly_equal = [](auto l, auto r) { return std::abs(l - r) <= (0.01 * std::abs(r)); };
    constexpr auto load_vec_and_info = [](auto&& path, auto&& group_name, auto&& dataset_name) {
        auto in = h5::file_t::open(path, H5F_ACC_RDONLY);
        auto group = in.open_group(group_name);

        return std::pair{fmt::format("{}:{}{}", path.filename().native(), group_name, dataset_name),
                         read_vector(group.get_id(), dataset_name)};
    };

    const auto [lhs_id, lhs] = load_vec_and_info(args.input_file, *args.lhs_group, *args.lhs_dataset);
    const auto [rhs_id, rhs]
      = load_vec_and_info(args.input_file_2 ? *args.input_file_2 : args.input_file, *args.rhs_group, *args.rhs_dataset);

    bool correct = true;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!nearly_equal(lhs[i], rhs[i])) {
            spdlog::info("{}: expected {}, got {}", i, lhs[i], rhs[i]);
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

int main(int argc, char* argv[]) try {
    auto app = structopt::app(brr::app_info.full_name, brr::app_info.version);
    auto arguments = app.parse<dim_cli>(argc, argv);
    spdlog::set_level(*arguments.log_level);

    if (arguments.store_matrix.has_value())
        return store_matrix(arguments.store_matrix);

    if (arguments.compare_results.has_value())
        return compare_results(arguments.compare_results);

    if (arguments.download.has_value())
        return download(arguments.download);

    if (arguments.generate_heatmap.has_value())
        return generate_heatmap(arguments.generate_heatmap);

    fmt::print("{}", app.help());

    return 0;
} catch (const std::exception& e) {
    spdlog::critical("exception thrown: {}", e.what());
    return 1;
}