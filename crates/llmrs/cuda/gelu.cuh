/*
GELU function declarations
*/
#ifndef GELU_CUH
#define GELU_CUH

#include "cuda_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for GELU functions
void gelu_forward(floatX* out, const floatX* inp, int N, cudaStream_t stream);
void gelu_backward_inplace(floatX* d_in_out, const floatX* inp, const int N, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // GELU_CUH
