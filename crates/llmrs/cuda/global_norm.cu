/*
Global norm, used in gradient clipping
*/
#include <assert.h>
#include <stddef.h>
#include <cuda_runtime_api.h>
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

template<class T>
__device__ float global_norm_squared_for_range(const T* data, size_t count) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_width = blockDim.x * gridDim.x;
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }
    // block-level reduce
    return blockReduce<warpReduceSum>(accumulator);
}

template<class T>
__global__ void global_norm_squared_kernel(float* out, const T* data, size_t count, ptrdiff_t stride) {
    float block_sum = global_norm_squared_for_range(data + blockIdx.y * stride, count);
    // each block accumulates its partial sum to out[out_index]
    // we want to avoid using atomic add here so we combine this kernel with another kernel call
    // that sums up the partial block sums
    if(threadIdx.x == 0) {
        size_t out_index = blockIdx.y * gridDim.x + blockIdx.x;
        out[out_index] = out[out_index] + block_sum;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher



template<typename T>
void global_norm_squared(float* out, const T* values, size_t count, ptrdiff_t stride, int num_slices, int max_num_block_sums, bool reset, cudaStream_t stream) {
    const int block_size = 512;
    // launch just enough blocks to fill the grid. deliberately no DIV_CEIL.
    // having one block less than possible is a tiny performance hit, having
    // one block too many is catastrophic, since it only can start once all the other
    // blocks finish. anyway, I think cuda_threads_per_SM should be a multiple of 512
    // on all gpus, so the division really is going to be exact.
    const int grid_size = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / block_size;
    assert(grid_size > 0);      // gives a better error than letting the call below fail

    const int gx = CEIL_DIV(grid_size, num_slices);
    const int gy = num_slices;

    assert(gx * gy < 1024);  // we want to later accumulate the block sums in a single block

    if (reset) {
        cudaCheck(cudaMemsetAsync(out, 0, max_num_block_sums * sizeof(float), stream));
    }
    global_norm_squared_kernel<<<dim3(gx, gy), block_size, 0, stream>>>(out, values, count, stride);
    cudaCheck(cudaGetLastError());
}


extern "C" {
    // Helper function determines the maximum number of block sums
    int get_max_num_block_sums(int* num_slices_all, int numel) {
        // NOTE: this needs to be kept in sync with `global_norm_squared` below.
        const int block_size = 512;
        const int grid_size = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / block_size;
        assert(grid_size > 0);
        int max_num_block_sums = 0;
        for (int i = 0; i < numel; i++) {
            int num_slices = num_slices_all[i];
            const int gx = CEIL_DIV(grid_size, num_slices);
            const int gy = num_slices;
            max_num_block_sums = max(max_num_block_sums, gx * gy);
        }

        return max_num_block_sums;
    }
    
    void global_norm_squared_float(float* out, const float* values, size_t count, ptrdiff_t stride, int num_slices, int max_num_block_sums, bool reset, cudaStream_t stream) {
        global_norm_squared<float>(out, values, count, stride, num_slices, max_num_block_sums, reset, stream);
    }

    void global_norm_squared_fp16(float* out, const half* values, size_t count, ptrdiff_t stride, int num_slices, int max_num_block_sums, bool reset, cudaStream_t stream) {
        global_norm_squared<half>(out, values, count, stride, num_slices, max_num_block_sums, reset, stream);
    }

    void global_norm_squared_bf16(float* out, const __nv_bfloat16* values, size_t count, ptrdiff_t stride, int num_slices, int max_num_block_sums, bool reset, cudaStream_t stream) {
        global_norm_squared<__nv_bfloat16>(out, values, count, stride, num_slices, max_num_block_sums, reset, stream);
    }
}