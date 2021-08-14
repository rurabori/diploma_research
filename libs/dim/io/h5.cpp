#include <dim/io/h5.h>

namespace dim::io::h5 {

auto read_vector(const H5::Group& group, const std::string& dataset_name) -> std::vector<double> {
    return detail::read_dataset<std::vector<double>>(group, dataset_name, H5::PredType::IEEE_F64LE,
                                                     H5::PredType::NATIVE_DOUBLE);
}

} // namespace dim::io::h5