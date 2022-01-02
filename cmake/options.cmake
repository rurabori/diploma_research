option(enable_petsc_benchmark "Enable petsc benchmark." OFF)
option(enable_mkl_benchmark "Enable mkl benchmark." OFF)
option(system_scientific_libs
       "Use system built scientific libs instead of conan packages." OFF)
option(
  invoke_conan
  "Invoke conan from cmake instead of requiring it to be ran before configuring cmake."
  ON)
