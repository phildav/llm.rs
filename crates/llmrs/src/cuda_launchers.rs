use cust::memory::DeviceMemory;
// Declaration of CUDA launcher we are using in the Rust code.
use cust::sys::CUstream;
use cust::sys::CUdeviceptr;
use cust::prelude::*;
use crate::common::FloatX;
use libc::{c_int, c_void};


#[allow(non_snake_case)]
pub fn k_fused_classifier(logits: &mut DeviceSlice<FloatX>, losses: &mut DeviceSlice<f32>, dloss: f32, targets: &DeviceSlice<i32>, B: usize, T: usize, V: usize, P: usize, write_dlogits: bool, stream: &Stream) {
    unsafe {
        fused_classifier(logits.as_raw_ptr() as *mut FloatX, losses.as_raw_ptr() as *mut f32, dloss, targets.as_raw_ptr() as *const i32, B as i32, T as i32, V as i32, P as i32, write_dlogits, stream.as_inner() as *mut c_void);
    }
}


pub fn global_norm_squared(result: DevicePointer<f32>, values: DevicePointer<FloatX>, count: usize, stride: isize, num_slices: i32, max_num_block_sums: i32, reset: bool, stream: CUstream) {
    unsafe {
        #[cfg(feature = "bf16")] global_norm_squared_bf16(result.as_raw(), values.as_raw(), count, stride, num_slices, max_num_block_sums, reset, stream);
        #[cfg(feature = "fp16")] global_norm_squared_fp16(result.as_raw(), values.as_raw(), count, stride, num_slices, max_num_block_sums, reset, stream);
        #[cfg(all(not(feature = "bf16"), not(feature = "fp16")))] global_norm_squared_float(result.as_raw(), values.as_raw(), count, stride, num_slices, max_num_block_sums, reset, stream);
    }
}

#[repr(C, align(16))]        // matches CUDA's int4 alignment/layout
#[derive(Clone, Copy, Default)]
pub struct Int4 { pub x: i32, pub y: i32, pub z: i32, pub w: i32 }




unsafe extern "C" {
    pub fn cublas_init();

    pub fn copy_and_cast(dst: CUdeviceptr, src: CUdeviceptr, n: usize, stride_dst: usize, stride_src: usize, grid_size: usize, num_layers: usize, stream: CUstream);
    pub fn adamw_update(params_memory: CUdeviceptr, master_params_memory: CUdeviceptr, grads_memory: CUdeviceptr, m_memory: CUdeviceptr, v_memory: CUdeviceptr, num_parameters: usize, w_stride: usize, g_stride: usize, s_stride: usize, num_slices: usize, learning_rate: f32, beta1: f32, beta2: f32, t: i32, eps: f32, weight_decay: f32, grad_scale: f32, seed: u32, stream: CUstream);
    pub fn init_from_master(params_memory: CUdeviceptr, master_params_memory: CUdeviceptr, num_parameters: usize, w_stride: usize, s_stride: usize, num_slices: usize, seed: u32, stream: CUstream);

    fn fused_classifier(logits: *mut FloatX, losses: *mut f32, dloss: f32, targets: *const i32, B: i32, T: i32, V: i32, P: i32, write_dlogits: bool, stream: *mut c_void);

    pub fn encoder_forward(out: CUdeviceptr, inp: CUdeviceptr, wte: CUdeviceptr, wpe: CUdeviceptr, B: i32, T: i32, C: i32, stream: CUstream);
    //pub fn encoder_backward(dwte: CUdeviceptr, dwpe: CUdeviceptr, scratch: CUdeviceptr, workload_indices: CUdeviceptr, bucket_info: CUdeviceptr, dout: CUdeviceptr, inp: CUdeviceptr, inputs_cpu: CUdeviceptr, B: i32, T: i32, C: i32, seed: u32, stream: CUstream);
    pub fn encoder_backward(
        dwte: *mut c_void,        // device floatX*
        dwpe: *mut c_void,        // device floatX*
        scratch: *mut c_void,     // device floatX*
        workload_indices: *mut c_int, // HOST int*
        bucket_info: *mut Int4,       // HOST int4*
        dout: *const c_void,      // device floatX*
        inp: *const c_void,       // device int* (GPU)
        inputs_cpu: *const c_int, // HOST const int*
        B: c_int, T: c_int, C: c_int,
        seed: u32,
        stream: *mut c_void       // cudaStream_t
    );

    pub fn layernorm_forward(out: CUdeviceptr, mean: CUdeviceptr, rstd: CUdeviceptr, inp: CUdeviceptr, weight: CUdeviceptr, bias: CUdeviceptr, B: i32, T: i32, C: i32, stream: CUstream);
    pub fn residual_forward(out: CUdeviceptr, inp1: CUdeviceptr, inp2: CUdeviceptr, N: i32, stream: CUstream);
    pub fn fused_residual_forward5(residual: CUdeviceptr, normed: CUdeviceptr, mean: CUdeviceptr, rstd: CUdeviceptr, inp1: CUdeviceptr, inp2: CUdeviceptr, weight: CUdeviceptr, bias: CUdeviceptr, N: i32, C: i32, stream: CUstream);
    pub fn layernorm_backward(dinp: CUdeviceptr, dweight: CUdeviceptr, dbias: CUdeviceptr, scratch: CUdeviceptr, dout: CUdeviceptr, inp: CUdeviceptr, weight: CUdeviceptr, mean: CUdeviceptr, rstd: CUdeviceptr, B: i32, T: i32, C: i32, stream: CUstream);
    
    // Matrix multiplication functions
    pub fn matmul_forward_cublaslt(out: CUdeviceptr, inp: CUdeviceptr, weight: CUdeviceptr, bias: CUdeviceptr, B: i32, T: i32, C: i32, OC: i32, stream: CUstream, pre_gelu: CUdeviceptr, gelu_fusion: i32);
    pub fn matmul_backward(dinp: CUdeviceptr, dweight: CUdeviceptr, dbias: CUdeviceptr, dout: CUdeviceptr, inp: CUdeviceptr, weight: CUdeviceptr, dbias_buffer: CUdeviceptr, B: i32, T: i32, C: i32, OC: i32, stream: CUstream, pre_gelu: CUdeviceptr, gelu_fusion: i32);
    
    // Attention functions
    pub fn attention_forward(out: CUdeviceptr, qkvr: CUdeviceptr, att: CUdeviceptr, inp: CUdeviceptr, B: i32, T: i32, C: i32, NH: i32, stream: CUstream);
    pub fn attention_backward(dinp: CUdeviceptr, dqkvr: CUdeviceptr, datt: CUdeviceptr, scratch: CUdeviceptr, dout: CUdeviceptr, qkvr: CUdeviceptr, att: CUdeviceptr, B: i32, T: i32, C: i32, NH: i32, stream: CUstream);

    // CUDA utils
    pub fn global_sum_deterministic_float(result: CUdeviceptr, values: CUdeviceptr, count: i32, stream: CUstream);
    pub fn global_sum_deterministic_fp16(result: CUdeviceptr, values: CUdeviceptr, count: i32, stream: CUstream);
    pub fn global_sum_deterministic_bf16(result: CUdeviceptr, values: CUdeviceptr, count: i32, stream: CUstream);
    
    // Global norm functions
    pub fn get_max_num_block_sums(num_slices_all: *const i32, numel: i32) -> i32;
    pub fn global_norm_squared_float(out: CUdeviceptr, values: CUdeviceptr, count: usize, stride: isize, num_slices: i32, max_num_block_sums: i32, reset: bool, stream: CUstream);
    pub fn global_norm_squared_fp16(out: CUdeviceptr, values: CUdeviceptr, count: usize, stride: isize, num_slices: i32, max_num_block_sums: i32, reset: bool, stream: CUstream);
    pub fn global_norm_squared_bf16(out: CUdeviceptr, values: CUdeviceptr, count: usize, stride: isize, num_slices: i32, max_num_block_sums: i32, reset: bool, stream: CUstream);

    // GELU functions
    pub fn gelu_forward(out: CUdeviceptr, inp: CUdeviceptr, N: i32, stream: CUstream);
    pub fn gelu_backward_inplace(d_in_out: CUdeviceptr, inp: CUdeviceptr, N: i32, stream: CUstream);
}