#include "dim/csr5_mpi/edge_sync.h"
#include "dim/csr5_mpi/output_range.h"
#include "dim/csr5_mpi/result_sync.h"
#include <dim/csr5_mpi/csr5_mpi.h>

#include <dim/io/h5.h>
#include <dim/mpi/mpi.h>
#include <mpi.h>

namespace dim::csr5_mpi {

auto csr5_partial::load(const std::filesystem::path& path, const std::string& group_name, MPI_Comm communicator)
  -> csr5_partial {
    using io::h5::load_csr5_partial;

    auto access = io::h5::plist_t::create(H5P_FILE_ACCESS);
    h5_try ::H5Pset_fapl_mpio(access.get_id(), communicator, MPI_INFO_NULL);

    auto in = io::h5::file_t::open(path, H5F_ACC_RDONLY, access);

    return {.communicator = communicator,
            .matrix = load_csr5_partial(in.open_group(group_name),
                                        {.idx = mpi::rank(communicator), .total_count = mpi::size(communicator)})};
}

auto csr5_partial::all_output_ranges() const noexcept -> std::vector<output_range_t> {
    const auto my_output_range = output_range();
    const auto comm_size = mpi::size(communicator);

    auto result = std::vector<output_range_t>(comm_size);
    ::MPI_Allgather(&my_output_range, 2, MPI_UINT32_T, result.data(), 2, MPI_UINT32_T, MPI_COMM_WORLD);

    return result;
}
auto csr5_partial::make_edge_sync() const noexcept -> edge_sync_t { return make_edge_sync(all_output_ranges()); }

auto csr5_partial::make_edge_sync(std::span<const output_range_t> output_ranges) const noexcept -> edge_sync_t {
    const auto current = mpi::rank(communicator);
    return edge_sync_t::create(output_range_t::syncs_downto(output_ranges, current), communicator);
}

auto csr5_partial::make_result_sync() const noexcept -> result_sync_t { return make_result_sync(all_output_ranges()); }
auto csr5_partial::make_result_sync(std::span<const output_range_t> output_ranges) const noexcept -> result_sync_t {
    return result_sync_t::create(output_ranges, communicator);
}

auto csr5_partial::make_sync() const noexcept -> sync_t { return make_sync(all_output_ranges()); }

auto csr5_partial::make_sync(std::span<const output_range_t> output_ranges) const noexcept -> sync_t {
    return {.edge_sync = make_edge_sync(output_ranges), .result_sync = make_result_sync(output_ranges)};
}
} // namespace dim::csr5_mpi