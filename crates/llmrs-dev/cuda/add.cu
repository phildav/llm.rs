#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add(const float *a, const float *b, float *out, size_t n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

extern "C" void add_cuda(const float *a, const float *b, float *out, size_t n) {
    
    add<<<(n+1023)/1024, 1024>>>(a, b, out, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaDeviceSynchronize();
}