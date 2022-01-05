#include <spdlog/spdlog.h>

#include <dim/simple_main.h>
#include <stdexcept>

#include "arguments.h"
#include "subcommands.h"
#include "version.h"

auto main_impl(const dim_cli& arguments) -> int {
    spdlog::set_level(*arguments.log_level);

    if (arguments.store_matrix.has_value())
        return store_matrix(arguments.store_matrix);

    if (arguments.compare_results.has_value())
        return compare_results(arguments.compare_results);

    if (arguments.download.has_value())
        return download(arguments.download);

    if (arguments.generate_heatmap.has_value())
        return generate_heatmap(arguments.generate_heatmap);

    throw std::invalid_argument{"No subcommand provided, please run with --help to see available subcommands."};
}
DIM_MAIN(dim_cli);
