#include <dim/io/h5.h>

namespace dim::io::h5 {

dataset_props_t::operator H5::DSetCreatPropList() const {
    H5::DSetCreatPropList prop_list{H5::DSetCreatPropList::DEFAULT};
    if (chunk_size)
        prop_list.setChunk(1, std::addressof(*chunk_size));

    if (compression_level)
        prop_list.setDeflate(*compression_level);

    return prop_list;
}

auto read_vector(const H5::Group& group, const std::string& dataset_name) -> std::vector<double> {
    return detail::read_dataset<std::vector<double>>(group, dataset_name, H5::PredType::IEEE_F64LE,
                                                     H5::PredType::NATIVE_DOUBLE);
}

} // namespace dim::io::h5