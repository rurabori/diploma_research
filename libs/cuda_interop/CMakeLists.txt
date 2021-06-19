find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)

brr_add_library(cui STATIC)
add_library(cui::cui ALIAS cui)
target_link_libraries(cui PUBLIC CUDA::cudart_static fmt::fmt)
