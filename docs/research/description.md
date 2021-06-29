# Description

+ review existing solutions for SpMV
+ review SpM storage formats (and distribution algos. for parallel version)
+ research possibility of paralelization of SpMV (via OMP, CUDA...)
+ implement OMP version of parallel SpMV
+ benchmark how different SpM structures (banded, diagonal, ...) perform.
+ research possibility of fusing matrix multiplication with vector multiplication (something like FMADD on GPUs).
+ consider optimizations for iterative solvers
(CG, BiCG).

# Description v2

+ implement single machine version.
+ research existing solutions for distributed SpMV.
+ implement distributed SpMV.
+ optimize distribution of data between nodes.
+ benchmark the distributed solution against 
+ discuss the benchmarks and viability of distributed solution.

# Description v3

1. Review existing approaches to distributed SPmV implementations. F.x. [1]
2. Implement distributed SpMV. 
3. Discuss possible optimizations of the implemented algorithm. 
4. Implement some of the discussed optimizations and measure the resulting speedup. 
5. Compare the performance of the implementation with similarly focused libraries as well as single machine implementation.
[1]J. Eckstein ,G. Matyasfalvi (2018) Efficient Distributed-Memory Parallel Matrix-Vector Multiplication with Wide or Tall Unstructured Sparse Matrices


