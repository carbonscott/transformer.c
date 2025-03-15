#ifndef CUDA_COMMON_MINIMAL_H
#define CUDA_COMMON_MINIMAL_H

#include <cuda_runtime.h>

// Precision setting (choose one)
// #define ENABLE_FP32
// #define ENABLE_FP16
#define ENABLE_BF16

// Precision settings
#if defined(ENABLE_FP32)
typedef float floatX;
#elif defined(ENABLE_FP16)
typedef half floatX;
#else // default to bfloat16
typedef __nv_bfloat16 floatX;
#endif

// CUDA error checking
// #define cudaCheck(err) { cudaError_t error = err; if (error != cudaSuccess) { printf("CUDA error: %s at line %d\n", cudaGetErrorString(error), __LINE__); exit(1); } }
#define cudaCheck(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// Division with ceiling
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#endif // CUDA_COMMON_MINIMAL_H
