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

    template<typename Container>
    auto read_dataset(h5::dataset_view_t dataset, h5::dataspace_view_t file_space, hsize_t count,
                      h5::type_view_t type) {
        Container values;
        values.resize(count);

        auto memspace = h5::dataspace_t::create(count);

        dataset.read(std::data(values), type, memspace, file_space);

        return values;
    }

    template<typename Container>
    auto load_known_range(h5::dataset_view_t dataset, hsize_t start, hsize_t count, h5::type_view_t type) {
        auto space = dataset.get_dataspace();

        space.select_hyperslab(start, count);

        return read_dataset<Container>(dataset, space, count, type);
    }

    auto read_csr5(h5::group_view_t matrix_group) {
        auto dataset = matrix_group.open_dataset("tile_desc");
        auto space = dataset.get_dataspace();
        auto dims = space.get_dim();

        const auto block_size = static_cast<hsize_t>(std::ceil(static_cast<double>(dims) / size));
        const hsize_t start = block_size * rank;
        const hsize_t count = std::min(block_size, dims - start);

        const auto& desc_type = tile_types_t::create();

        auto&& tile_desc
          = load_known_range<decltype(mat::csr5<double>::tile_desc)>(dataset, start, count, desc_type.in_memory);

        auto tile_size = tile_desc.size();
        auto&& tile_ptr = load_known_range<decltype(mat::csr5<double>::tile_ptr)>(matrix_group.open_dataset("tile_ptr"),
                                                                                  start, count + 1, H5T_NATIVE_UINT32);

        const auto row_start = tile_ptr.front();
        const auto row_end = tile_ptr.back();

        const auto row_ptr_dataset = matrix_group.open_dataset("row_ptr");
        const auto dimensions
          = mat::dimensions_t{.rows = static_cast<uint32_t>(row_ptr_dataset.get_dataspace().get_dim() - 1),
                              .cols = h5::detail::read_scalar_datatype<uint32_t>(matrix_group, "column_count",
                                                                                 H5T_STD_U32LE, H5T_NATIVE_UINT32)};

        // +1 because indices are 0 based, +1 because we need to load data upto row at last + 1.
        auto&& row_ptr = load_known_range<decltype(mat::csr5<double>::row_ptr)>(
          row_ptr_dataset, row_start, row_end - row_start + 2, H5T_NATIVE_UINT32);

        auto&& tile_desc_offset_ptr = load_known_range<decltype(mat::csr5<double>::tile_desc_offset_ptr)>(
          matrix_group.open_dataset("tile_desc_offset_ptr"), start, count + 1, H5T_NATIVE_UINT32);

        const auto tile_desc_offset_ptr_start = tile_desc_offset_ptr.front();
        const auto tile_desc_offset_ptr_end = tile_desc_offset_ptr.back();

        auto&& tile_desc_offset = load_known_range<decltype(mat::csr5<double>::tile_desc_offset)>(
          matrix_group.open_dataset("tile_desc_offset"), tile_desc_offset_ptr_start,
          tile_desc_offset_ptr_end - tile_desc_offset_ptr_start, H5T_NATIVE_UINT32);

        auto val_start = row_ptr.front();
        auto val_end = row_ptr.back();

        return mat::csr5<double>{
          .dimensions = dimensions,
          .vals = load_known_range<decltype(mat::csr5<double>::vals)>(matrix_group.open_dataset("vals"), val_start,
                                                                      val_end - val_start, H5T_NATIVE_DOUBLE),
          .col_idx = load_known_range<decltype(mat::csr5<double>::col_idx)>(
            matrix_group.open_dataset("col_idx"), val_start, val_end - val_start, H5T_NATIVE_UINT32),
          .row_ptr = std::move(row_ptr),
          .tile_count = tile_size,
          .tile_ptr = std::move(tile_ptr),
          .tile_desc = std::move(tile_desc),
          .tile_desc_offset_ptr = std::move(tile_desc_offset_ptr),
          .tile_desc_offset = std::move(tile_desc_offset),
          .skip_tail = (rank != size - 1)};
    }
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

    // step 1. do all of the full tiles
    csr5.spmv(x, y);
    // step 2. do calibrators
    // step 3.a tail partition for the last tile.
    // step 3.b broadcast calibrators to the left neighbour

    auto&& first_node_row = y[csr5.tile_ptr.front()];
    auto&& last_node_row = y[csr5.tile_ptr.back()];

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
    auto write_end = csr5.tile_ptr.back();

    auto filespace = dataset.get_dataspace();
    filespace.select_hyperslab(write_start, write_end - write_start);

    auto span = std::span{y}.subspan(write_start, write_end - write_start);
    dataset.write(span.data(), H5T_NATIVE_DOUBLE, h5::dataspace_t::create(hsize_t{span.size()}), filespace, xfer_pl);

    return 0;
} catch (const std::exception& e) {
    fmt::print(stderr, "Exception thrown: {}\n", e.what());
    return 1;
}
