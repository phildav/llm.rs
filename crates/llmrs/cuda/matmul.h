/*
Matrix multiplication function declarations
*/
#ifndef MATMUL_H
#define MATMUL_H

#include "cuda_common.h"

// Forward declarations for matrix multiplication functions
void matmul_cublaslt(floatX* d, const floatX* a, const floatX* b, const floatX* bias,
                     int m, int n, int k, cudaStream_t stream=0, bool transA=true, bool transB=false,
                     int batch_count=0, size_t strideA=0, size_t strideB=0, size_t strideOut=0,
                     bool accumulate=false, floatX* pre_gelu=NULL, bool backward=false);

// Small wrapper around matmul_cublaslt for the forward pass (keeping historical order of arguments)
void matmul_forward(floatX* out, const floatX* inp, const floatX* weight, const floatX* bias,
                    int OC, int B, int T, int C, cudaStream_t stream, floatX* pre_gelu=NULL);

// Small wrapper around matmul_cublaslt for the backward pass
void matmul_backward(floatX* dinp, floatX* dweight, floatX* dbias, const floatX* dout,
                     const floatX* inp, const floatX* weight, int OC, int B, int T, int C, cudaStream_t stream);

#endif // MATMUL_H
