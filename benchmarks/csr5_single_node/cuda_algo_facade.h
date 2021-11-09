#ifndef APPS_CONJUGATE_GRADIENT_CUDA_ALGO_FACADE
#define APPS_CONJUGATE_GRADIENT_CUDA_ALGO_FACADE

void bench_csr5_cuda(int rows, int cols, int non_zero, int* row_start_offsets, int* col_indices, double* values,
                     double* rhs, double* output);

#endif /* APPS_CONJUGATE_GRADIENT_CUDA_ALGO_FACADE */
