#ifndef ERROR_CHECK_CUH
#define ERROR_CHECK_CUH

#include <cuda_runtime.h>
#include <curand.h> // Include cuRAND header
#include <stdio.h>  // For fprintf
#include <stdlib.h> // For exit, EXIT_FAILURE

// cuda API error checking
#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

// cuRAND API error checking
#define CURAND_CHECK(call)                                                    \
    {                                                                         \
        curandStatus_t err = call;                                            \
        if (err != CURAND_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuRAND error at %s:%d - %d\n", __FILE__,         \
                    __LINE__, err);                                           \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }
#endif
