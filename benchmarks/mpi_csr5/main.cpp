#include <dim/io/h5.h>
#include <dim/io/h5/plist.h>
#include <dim/mpi/mpi.h>
#include <dim/simple_main.h>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/stopwatch.h>

#include <filesystem>
#include <mpi.h>
#include <optional>
#include <string>

namespace fs = std::filesystem;
namespace h5 = dim::io::h5;

struct arguments_t
{
    fs::path input_file;
    fs::path output_file;
    std::optional<std::string> input_group{"A"};
    std::optional<std::string> output_dataset{"Y"};
};
STRUCTOPT(arguments_t, input_file, output_file, input_group, output_dataset);

template<const auto& Fun, typename Ty>
auto mpi_query_com(MPI_Comm comm) {
    Ty result;
    if (const auto status = Fun(comm, &result); status != MPI_SUCCESS)
        throw std::runtime_error{fmt::format("MPI failed with status code: {}", status)};

    return result;
}

auto mpi_rank() { return static_cast<size_t>(mpi_query_com<::MPI_Comm_rank, int>(MPI_COMM_WORLD)); }

auto mpi_size() { return static_cast<size_t>(mpi_query_com<::MPI_Comm_size, int>(MPI_COMM_WORLD)); }

auto read_matrix(const fs::path& path, const std::string& name) {
    auto access = h5::plist_t::create(H5P_FILE_ACCESS);
    h5_try ::H5Pset_fapl_mpio(access.get_id(), MPI_COMM_WORLD, MPI_INFO_NULL);

    auto in = h5::file_t::open(path, H5F_ACC_RDONLY, access);

    return h5::load_csr5_partial(in.open_group(name), {.idx = mpi_rank(), .total_count = mpi_size()});
}

auto main_impl(const arguments_t& args) {
    spdlog::stopwatch sw;
    const auto csr5 = read_matrix(args.input_file, *args.input_group);
    spdlog::info("loading matrix took {}s", sw);

    const auto first_tile_row = csr5.tile_ptr.front().idx();
    const auto last_tile_row = csr5.tile_ptr.back().idx();
    const auto global_last_row = csr5.skip_tail ? last_tile_row : (csr5.dimensions.rows - 1);
    // the range is closed from both sides hence the +1. f.x. [0,0] has 1 element etc.
    const auto this_node_elements = global_last_row - first_tile_row + 1;

    std::vector<double> x(csr5.dimensions.cols, 1.);
    std::vector<double> y(this_node_elements, 0.);
    auto calibrator = csr5.allocate_calibrator();

    const auto rank = mpi_rank();
    const auto size = mpi_size();

    sw.reset();
    csr5.spmv({.x = x, .y = y, .calibrator = calibrator});
    spdlog::info("spmv took {}s", sw);

    auto&& first_node_row = y.front();
    auto&& last_node_row = y.back();

    sw.reset();
    if (rank == 0) {
        // we're the main node.
        double sum{};
        ::MPI_Recv(&sum, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        last_node_row += sum;
    } else if (rank == size - 1) {
        // we're in the last node.
        ::MPI_Send(&first_node_row, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
    } else {
        // we're in the middle node.
        double sum{};
        ::MPI_Sendrecv(&first_node_row, 1, MPI_DOUBLE, rank - 1, 0, &sum, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD,
                       MPI_STATUS_IGNORE);
        last_node_row += sum;
    }
    spdlog::info("synchronizing edge results took {}s", sw);

    auto access = h5::plist_t::create(H5P_FILE_ACCESS);
    h5_try ::H5Pset_fapl_mpio(access.get_id(), MPI_COMM_WORLD, MPI_INFO_NULL);

    auto out_file = h5::file_t::create(args.output_file, H5F_ACC_TRUNC, h5::plist_t::defaulted(), access);
    auto data_creation_props = h5::plist_t::create(H5P_DATASET_CREATE);

    const auto fill = double{0.};
    h5_try ::H5Pset_fill_value(data_creation_props.get_id(), H5T_NATIVE_DOUBLE, &fill);
    h5_try ::H5Pset_fill_time(data_creation_props.get_id(), H5D_FILL_TIME_IFSET);

    auto dataset
      = out_file.create_dataset(*args.output_dataset, H5T_IEEE_F64LE,
                                h5::dataspace_t::create(hsize_t{csr5.dimensions.rows}), h5::plist_t::defaulted());

    const auto is_main_node = rank == 0;

    auto write_start = first_tile_row + !is_main_node;

    const auto to_write = std::span{y}.subspan(!is_main_node);
    spdlog::info("writing elements [{},{}] {} elems", write_start, global_last_row, to_write.size());

    auto filespace = dataset.get_dataspace();
    filespace.select_hyperslab(write_start, to_write.size());

    sw.reset();

    auto xfer_pl = h5::plist_t::create(H5P_DATASET_XFER);
    h5_try ::H5Pset_dxpl_mpio(xfer_pl.get_id(), H5FD_MPIO_COLLECTIVE);

    dataset.write(to_write.data(), H5T_NATIVE_DOUBLE, h5::dataspace_t::create(hsize_t{to_write.size()}), filespace,
                  xfer_pl);

    spdlog::info("writing to output file took {}s", sw);

    return 0;
}

int main(int argc, char* argv[]) try {
    auto app = structopt ::app(brr ::app_info.full_name, brr ::app_info.version);
    dim::mpi::ctx mpi_ctx{argc, argv};
    spdlog::set_default_logger(spdlog::stdout_color_mt(fmt::format("mpi_csr5 (n{:02})", mpi_rank())));

    return main_impl(app.parse<arguments_t>(argc, argv));
} catch (const structopt ::exception& e) {
    fmt::print(stderr, "{}", e.help());
    spdlog::critical("{}", e.what());
    return 1;
} catch (const std ::exception& e) {
    spdlog::critical("{}", e.what());
    return 2;
}
