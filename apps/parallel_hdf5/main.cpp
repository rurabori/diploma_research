#include "dim/io/h5/dataset.h"
#include "dim/io/h5/dataspace.h"
#include "dim/io/h5/group.h"
#include "dim/io/h5/plist.h"
#include "dim/io/h5/type.h"
#include "dim/mat/storage_formats/base.h"
#include <H5Fpublic.h>
#include <H5Ppublic.h>
#include <H5Spublic.h>
#include <H5Tpublic.h>
#include <cstdint>
#include <dim/io/h5.h>
#include <dim/mpi/mpi.h>
#include <filesystem>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <hdf5.h>
#include <initializer_list>
#include <mpi.h>
#include <stdexcept>

namespace h5 = dim::io::h5;
namespace mat = dim::mat;

template<const auto& Fun, typename Ty>
auto mpi_query_com(MPI_Comm comm) {
    Ty result;
    if (const auto status = Fun(comm, &result); status != MPI_SUCCESS)
        throw std::runtime_error{fmt::format("MPI failed with status code: {}", status)};

    return result;
}

struct mpi_h5_reader_t
{
    const int size;
    const int rank;

    struct tile_types_t
    {
        static constexpr hsize_t array_size = 4;

        h5::type_t on_disk{h5::type_t::create_array(H5T_NATIVE_UINT32, array_size)};
        h5::type_t in_memory{h5::type_t::create_array(H5T_STD_U32LE, array_size)};

        static auto create() -> tile_types_t& {
            static tile_types_t tile_types;
            return tile_types;
        }
    };

    explicit mpi_h5_reader_t(MPI_Comm comm = MPI_COMM_WORLD)
      : size{mpi_query_com<::MPI_Comm_size, int>(comm)}, rank{mpi_query_com<::MPI_Comm_rank, int>(comm)} {}

    explicit mpi_h5_reader_t(int size_, int rank_) : size{size_}, rank{rank_} {}

    auto read_csr5(h5::group_view_t matrix_group) { return h5::load_csr5_partial(matrix_group, rank, size); }
};

int main(int argc, char* argv[]) try {
    dim::mpi::ctx mpi_ctx{argc, argv};

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Info info = MPI_INFO_NULL;

    mpi_h5_reader_t reader{comm};

    auto access = h5::plist_t::create(H5P_FILE_ACCESS);
    h5_try ::H5Pset_fapl_mpio(access.get_id(), comm, info);

    auto file = h5::file_t::open("nv2.csr5.h5", H5F_ACC_RDONLY, access);

    auto matrix_group = file.open_group("A");

    auto csr5 = reader.read_csr5(matrix_group);

    std::vector<double> x(csr5.dimensions.cols, 1.);
    std::vector<double> y(csr5.dimensions.rows, 0.);
    auto calibrator = csr5.allocate_calibrator();

    csr5.spmv({.x = x, .y = y, .calibrator = calibrator});

    auto&& first_node_row = y[0];
    auto&& last_node_row = y[(csr5.skip_tail ? csr5.tile_ptr.back() : csr5.dimensions.rows) - csr5.tile_ptr.front()];

    if (reader.rank == 0) {
        // we're the main node.
        double sum{};
        ::MPI_Recv(&sum, 1, MPI_DOUBLE, 1, 0, comm, MPI_STATUS_IGNORE);
        last_node_row += sum;
    } else if (reader.rank == reader.size - 1) {
        // we're in the last node.
        ::MPI_Send(&first_node_row, 1, MPI_DOUBLE, reader.rank - 1, 0, comm);
    } else {
        // we're in the middle node.
        double sum{};
        ::MPI_Sendrecv(&first_node_row, 1, MPI_DOUBLE, reader.rank - 1, 0, &sum, 1, MPI_DOUBLE, reader.rank + 1, 0,
                       comm, MPI_STATUS_IGNORE);
        last_node_row += sum;
    }

    auto out_file = h5::file_t::create("res.h5", H5F_ACC_TRUNC, h5::plist_t::defaulted(), access);
    auto dataset = out_file.create_dataset("Y", H5T_IEEE_F64LE, h5::dataspace_t::create(hsize_t{csr5.dimensions.rows}));

    /*
     * Create property list for collective dataset write.
     */
    auto xfer_pl = h5::plist_t::create(H5P_DATASET_XFER);
    h5_try ::H5Pset_dxpl_mpio(xfer_pl.get_id(), H5FD_MPIO_COLLECTIVE);

    auto write_start = csr5.tile_ptr.front() + bool(reader.rank);
    auto write_end = csr5.skip_tail ? csr5.tile_ptr.back() : (csr5.dimensions.rows - 1);

    fmt::print("{}: [{}, {}]\n", reader.rank, write_start, write_end);

    auto filespace = dataset.get_dataspace();
    filespace.select_hyperslab(write_start, write_end - write_start + 1);

    auto span = std::span{y}.subspan(bool(reader.rank), write_end - write_start + 1);
    dataset.write(span.data(), H5T_NATIVE_DOUBLE, h5::dataspace_t::create(hsize_t{span.size()}), filespace, xfer_pl);

    return 0;
} catch (const std::exception& e) {
    fmt::print(stderr, "Exception thrown: {}\n", e.what());
    return 1;
}
