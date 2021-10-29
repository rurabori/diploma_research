#include <dim/io/h5.h>
#include <dim/mpi/mpi.h>
#include <fmt/format.h>

#include <hdf5.h>

int main(int argc, char* argv[]) try {
    namespace h5 = dim::io::h5;

    dim::mpi::ctx mpi_ctx{argc, argv};

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Info info = MPI_INFO_NULL;

    int mpi_size{};
    MPI_Comm_size(comm, &mpi_size);
    int mpi_rank{};
    MPI_Comm_rank(comm, &mpi_rank);

    auto access = h5::plist_t::create(H5P_FILE_ACCESS);
    h5_try ::H5Pset_fapl_mpio(access.get_id(), comm, info);

    auto file = h5::file_t::create("test.h5", H5F_ACC_TRUNC, h5::plist_t::defaulted(), access);

    return 0;
} catch (const std::exception& e) {
    fmt::print(stderr, "Exception thrown: {}\n", e.what());
    return 1;
}
