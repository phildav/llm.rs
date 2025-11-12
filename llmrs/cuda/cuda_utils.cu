#include "cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>

// ----------------------------------------------------------------------------
// DType support - non-template implementations

// Given a datatype enum, returns the underlying number of bytes
// for a scalar of that type
size_t sizeof_dtype(DType type) {
    switch (type) {
        case DType::FP32:
            return sizeof(float);
        case DType::FP16:
            return sizeof(half);
        case DType::BF16:
            return sizeof(nv_bfloat16);
        default: // handle or get compiler warning
            fprintf(stderr, "Unknown datatype\n");
            exit(EXIT_FAILURE);
    }
}

DType dtype_of(float* f) { return DType::FP32; }
DType dtype_of(nv_bfloat16 * f) { return DType::BF16; }
DType dtype_of(half * f) { return DType::FP16; }

// Template specializations for cast_value
template<>
__device__ inline float cast_value<float, float>(float val) {
    return val;
}

template<>
__device__ inline float cast_value<float, half>(half val) {
    return __half2float(val);
}

template<>
__device__ inline float cast_value<float, __nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}


// ----------------------------------------------------------------------------
// Global kernel implementations

// Performs a _deterministic_ sum reduction. determinism is achieved by requiring that only
// a single block be used.
template<class Float>
__global__ void global_sum_single_block_kernel(float* result, const Float* values, size_t count) {
    assert(gridDim.x == 1);     // only a single block!
    float thread_sum = 0;
    for(size_t index = threadIdx.x; index < count; index += blockDim.x) {
        thread_sum += (float)values[index];
    }

    float reduction = blockReduce<warpReduceSum>(thread_sum, true);
    if(threadIdx.x == 0) {
        *result = reduction;
    }
}

template<class Float>
void global_sum_deterministic(float* result, const Float* values, int count, cudaStream_t stream) {
    global_sum_single_block_kernel<<<1, 1024, 0, stream>>>(result, values, count);
    cudaCheck(cudaGetLastError());
}

extern "C" {

    void copy_and_cast(float* dst, const floatX* src, size_t n, size_t stride_dst, size_t stride_src, size_t grid_size, size_t num_layers, CUstream stream) {
        /// Copy and cast from src to dst.
        /// n is the number of elements to copy per layer
        /// stride_dst and stride_src the strides per layer (in number of elements)

        // launch block of 512 threads. We have (shard.size / 512) blocks (gridDim.x)
        // times num_layers (gridDim.y)
        copy_and_cast_kernel<<<dim3(grid_size, num_layers), 512, 0, stream>>>(dst, src, n, stride_dst, stride_src);
        cudaCheck(cudaGetLastError());
    }

    // Explicit template instantiations for common types
    void global_sum_deterministic_float(float* result, const float* values, int count, cudaStream_t stream) {
        global_sum_deterministic<float>(result, values, count, stream);
    }

    void global_sum_deterministic_fp16(float* result, const half* values, int count, cudaStream_t stream) {
        global_sum_deterministic<half>(result, values, count, stream);
    }

    void global_sum_deterministic_bf16(float* result, const __nv_bfloat16* values, int count, cudaStream_t stream) {
        global_sum_deterministic<__nv_bfloat16>(result, values, count, stream);
    }

}