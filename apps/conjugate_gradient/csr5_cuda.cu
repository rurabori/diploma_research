#include "cuda_algo_facade.h"
#include <anonymouslib_cuda.cuh>
#include <cuda_interop/annotations.h>
#include <cuda_interop/memory.h>
#include <fmt/format.h>

__host__ void bench_csr5_cuda(int rows, int cols, int non_zero, int* row_start_offsets, int* col_indices,
                              double* values, double* rhs, double* output) {
    // set device
    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, device_id);

    fmt::print("Device [{}] {}, @{}MHz\n", device_id, deviceProp.name, deviceProp.clockRate * 1e-3f);

    auto d_row_start_offsets = cui::device_create(row_start_offsets, rows + 1);
    auto d_col_indices = cui::device_create(col_indices, non_zero);
    auto d_values = cui::device_create(values, non_zero);

    anonymouslibHandle<int, unsigned int, double> A(rows, cols);
    A.inputCSR(non_zero, d_row_start_offsets.get(), d_col_indices.get(), d_values.get());

    auto d_x = cui::device_create(rhs, cols);
    A.setX(d_x.get()); // you only need to do it once!

    A.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA);

    anonymouslib_timer asCSR5_timer;
    asCSR5_timer.start();
    A.asCSR5();
    fmt::print("CSR->CSR5 time = {}ms.\n", asCSR5_timer.stop());

    auto d_y = cui::alloc<double>(rows);
    anonymouslib_timer spmv_timer;
    spmv_timer.start();
    A.spmv(1.0, d_y.get());
    fmt::print("SpMV time = {}ms.\n", spmv_timer.stop());

    cui::memcpy(output, d_y.get(), rows, cudaMemcpyDeviceToHost);
}