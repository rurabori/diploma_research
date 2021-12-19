#ifndef INCLUDE_DIM_OPT
#define INCLUDE_DIM_OPT

#define DIM_STRINGIFY(a) #a

#ifdef __GNUC__

#define DIM_UNROLL_N(n) _Pragma(DIM_STRINGIFY(GCC unroll(n)))
#define DIM_UNROLL DIM_UNROLL_N(4)

#else

#define DIM_UNROLL_N(n) _Pragma(DIM_STRINGIFY(OMP unroll(n)))
#define DIM_UNROLL _Pragma(DIM_STRINGIFY(OMP unroll)

#endif

#endif /* INCLUDE_DIM_OPT */
