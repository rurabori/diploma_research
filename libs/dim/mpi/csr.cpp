#include <dim/mpi/csr.h>

#include <dim/io/h5.h>
#include <dim/mpi/mpi.h>

namespace dim::mpi {
auto load_csr_partial(const std::filesystem::path& path, const std::string& group_name, MPI_Comm communicator)
  -> mat::csr_partial_t<> {
    auto access = io::h5::plist_t::create(H5P_FILE_ACCESS);
    h5_try ::H5Pset_fapl_mpio(access.get_id(), communicator, MPI_INFO_NULL);

    auto in = io::h5::file_t::open(path, H5F_ACC_RDONLY, access);

    return io::h5::load_csr_partial(in.open_group(group_name),
                                    {.idx = mpi::rank(communicator), .total_count = mpi::size(communicator)});
}

} // namespace dim::mpi