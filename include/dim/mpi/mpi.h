#pragma once

#include <mpi.h>

namespace dim::mpi {

struct ctx
{
    ctx(int& argc, char**& argv);
    ~ctx();
};

} // namespace dim::mpi
