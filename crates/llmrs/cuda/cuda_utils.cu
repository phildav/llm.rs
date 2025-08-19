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

template<class Float>
void global_sum_deterministic(float* result, const Float* values, int count, cudaStream_t stream) {
    global_sum_single_block_kernel<<<1, 1024, 0, stream>>>(result, values, count);
    cudaCheck(cudaGetLastError());
}

// Explicit template instantiations for common types
template void global_sum_deterministic<float>(float* result, const float* values, int count, cudaStream_t stream);
template void global_sum_deterministic<half>(float* result, const half* values, int count, cudaStream_t stream);
template void global_sum_deterministic<__nv_bfloat16>(float* result, const __nv_bfloat16* values, int count, cudaStream_t stream);

extern "C" {

    void copy_and_cast(float* dst, const float* src, size_t n, size_t stride_dst, size_t stride_src, size_t grid_size, size_t num_layers, CUstream stream) {
        copy_and_cast_kernel<<<dim3(grid_size, num_layers), 512, 0, stream>>>(dst, src, n, stride_dst, stride_src);
        cudaCheck(cudaGetLastError());
    }

}