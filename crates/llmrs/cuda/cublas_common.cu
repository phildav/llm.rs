/*
cuBLAS global variable definitions
*/
#include "cuda_common.h"
#include "cublas_common.h"
#include <cublas_v2.h>
#include <cublasLt.h>

// Define the global variables declared as extern in cublas_common.h
const size_t cublaslt_workspace_size = 32 * 1024 * 1024;
void* cublaslt_workspace = NULL;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;



extern "C" {

    void cublas_init() {
        int deviceIdx = 0;
        cudaCheck(cudaSetDevice(deviceIdx));
        cudaGetDeviceProperties(&deviceProp, deviceIdx);
        printf("[cublas_init] Device %d: %s\n", deviceIdx, deviceProp.name);
        
    
        // setup cuBLAS and cuBLASLt
        cublasCheck(cublasCreate(&cublas_handle));
        cublasCheck(cublasLtCreate(&cublaslt_handle));
        // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
        int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
        printf("enable_tf32: %d\n", enable_tf32);
        cublas_compute = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
        cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
        cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
        // setup the (global) cuBLASLt workspace
        cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));
        fflush(stdout);
    }

}