#include <dim/mpi/mpi.h>
#include <fmt/format.h>

#include <vector>
#include <algorithm>
#include <tuple>

#include <H5Cpp.h>
#include <H5DataSet.h>
#include <H5DataSpace.h>
#include <H5File.h>
#include <H5FloatType.h>
#include <H5PredType.h>
#include <H5public.h>

int main(int argc, char* argv[]) {
    // dim::mpi::ctx mpi{argc, argv};

    // int world_size = 0;
    // ::MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // open+trunc
    H5::H5File file{argv[1], H5F_ACC_TRUNC};

    const hsize_t dims[] = {4, 4};
    H5::DataSpace dataspace{2, std::data(dims)};

    std::vector<double> test(16, 0.);
    std::ranges::generate(test, [ctr = 0]() mutable { return static_cast<double>(ctr++); });

    H5::FloatType data_type{H5::PredType::NATIVE_DOUBLE};
    data_type.setOrder(H5T_ORDER_LE);

    auto dataset = file.createDataSet(argv[1], data_type, dataspace);

    dataset.write(test.data(), H5::PredType::NATIVE_DOUBLE);

    // int world_rank = 0;
    // ::MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // fmt::print("Size: {} Rank:{} \n", world_size, world_rank);

    return 0;
}