#include <dim/io/h5/plist.h>

namespace dim::io::h5 {
plist_t plist_t::create(hid_t plist_class) { return plist_t{::H5Pcreate(plist_class)}; }
} // namespace dim::io::h5
