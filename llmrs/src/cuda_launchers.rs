use cust::memory::{DeviceMemory, DeviceCopy};
// Declaration of CUDA launcher we are using in the Rust code.
use cust::sys::CUstream;
use cust::sys::CUdeviceptr;
use cust::prelude::*;
use crate::common::FloatX;
use libc::{c_int, c_void};

#[repr(C, align(16))]        // matches CUDA's int4 alignment/layout
#[derive(Clone, Copy, Default, DeviceCopy)]
pub struct Int4 { pub x: i32, pub y: i32, pub z: i32, pub w: i32 }


#[allow(non_snake_case)]
pub fn k_copy_and_cast(dst: &mut DevicePointer<f32>, src: &DevicePointer<FloatX>, n: usize, stride_dst: usize, stride_src: usize, grid_size: usize, num_layers: usize, stream: &Stream) {
    unsafe {
        copy_and_cast(dst.as_raw() as *mut f32, src.as_raw() as *const FloatX, n, stride_dst, stride_src, grid_size, num_layers, stream.as_inner() as CUstream);
    }
}

#[allow(non_snake_case)]
pub fn k_adamw_update(params_memory: &mut DevicePointer<FloatX>, master_params_memory: &DevicePointer<f32>, grads_memory: &DevicePointer<FloatX>, m_memory: &DevicePointer<f32>, v_memory: &DevicePointer<f32>, num_parameters: usize, w_stride: usize, g_stride: usize, s_stride: usize, num_slices: usize, learning_rate: f32, beta1: f32, beta2: f32, t: i32, eps: f32, weight_decay: f32, grad_scale: f32, seed: u32, stream: &Stream) {
    unsafe {
        adamw_update(params_memory.as_raw() as *mut FloatX, master_params_memory.as_raw() as *const f32, grads_memory.as_raw() as *const FloatX, m_memory.as_raw() as *const f32, v_memory.as_raw() as *const f32, num_parameters, w_stride, g_stride, s_stride, num_slices, learning_rate, beta1, beta2, t, eps, weight_decay, grad_scale, seed, stream.as_inner() as CUstream);
    }
}

#[allow(non_snake_case)]
pub fn k_init_from_master(params_memory: &mut DevicePointer<FloatX>, master_params_memory: &DevicePointer<f32>, num_parameters: usize, w_stride: usize, s_stride: usize, num_slices: usize, seed: u32, stream: &Stream) {
    unsafe {
        init_from_master(params_memory.as_raw() as *mut FloatX, master_params_memory.as_raw() as *const f32, num_parameters, w_stride, s_stride, num_slices, seed, stream.as_inner() as CUstream);
    }
}

#[allow(non_snake_case)]
pub fn k_fused_classifier(logits: &mut DeviceSlice<FloatX>, losses: &mut DeviceSlice<f32>, dloss: f32, targets: &DeviceSlice<i32>, B: usize, T: usize, V: usize, P: usize, write_dlogits: bool, stream: &Stream) {
    unsafe {
        fused_classifier(logits.as_raw_ptr() as *mut FloatX, losses.as_raw_ptr() as *mut f32, dloss, targets.as_raw_ptr() as *const i32, B as i32, T as i32, V as i32, P as i32, write_dlogits, stream.as_inner() as *mut c_void);
    }
}

#[allow(non_snake_case)]
pub fn k_encoder_forward(out: &mut DeviceSlice<FloatX>, inp: &DeviceSlice<i32>, wte: &DeviceSlice<FloatX>, wpe: &DeviceSlice<FloatX>, B: usize, T: usize, C: usize, stream: &Stream) {
    unsafe {
        encoder_forward(out.as_raw_ptr() as *mut FloatX, inp.as_raw_ptr() as *const i32, wte.as_raw_ptr() as *const FloatX, wpe.as_raw_ptr() as *const FloatX, B as i32, T as i32, C as i32, stream.as_inner() as CUstream);
    }
}

#[allow(non_snake_case)]
pub fn k_encoder_backward(dwte: &mut DeviceSlice<FloatX>, dwpe: &mut DeviceSlice<FloatX>, scratch: &mut DeviceSlice<FloatX>, workload_indices: &mut [i32], bucket_info: &mut [Int4], dout: &DeviceSlice<FloatX>, inp: &DeviceSlice<i32>, inputs_cpu: &[i32], B: usize, T: usize, C: usize, seed: u32, stream: &Stream) {
    unsafe {
        encoder_backward(dwte.as_raw_ptr() as *mut FloatX, dwpe.as_raw_ptr() as *mut FloatX, scratch.as_raw_ptr() as *mut FloatX,
            workload_indices.as_mut_ptr(), bucket_info.as_mut_ptr(),
            dout.as_raw_ptr() as *const FloatX, inp.as_raw_ptr() as *const i32, inputs_cpu.as_ptr() as *const i32,
            B as i32, T as i32, C as i32, seed, stream.as_inner() as CUstream);
    }
}

#[allow(non_snake_case)]
pub fn k_layernorm_forward(out: &mut DeviceSlice<FloatX>, mean: &mut DeviceSlice<f32>, rstd: &mut DeviceSlice<f32>,
                            inp: &DeviceSlice<FloatX>, weight: &DeviceSlice<FloatX>, bias: &DeviceSlice<FloatX>,
                            B: usize, T: usize, C: usize, stream: &Stream) {
    unsafe {
        layernorm_forward(out.as_raw_ptr() as *mut FloatX, mean.as_raw_ptr() as *mut f32, rstd.as_raw_ptr() as *mut f32,
            inp.as_raw_ptr() as *const FloatX, weight.as_raw_ptr() as *const FloatX, bias.as_raw_ptr() as *const FloatX,
            B as i32, T as i32, C as i32, stream.as_inner() as CUstream);
    }
}

#[allow(non_snake_case)]
pub fn k_residual_forward(out: &mut DeviceSlice<FloatX>, inp1: &DeviceSlice<FloatX>, inp2: &DeviceSlice<FloatX>,
                            N: usize, stream: &Stream) {
    unsafe {
        residual_forward(out.as_raw_ptr() as *mut FloatX, inp1.as_raw_ptr() as *const FloatX, inp2.as_raw_ptr() as *const FloatX,
            N as i32, stream.as_inner() as CUstream);
    }
}

#[allow(non_snake_case)]
pub fn k_fused_residual_forward5(residual: &mut DeviceSlice<FloatX>, normed: &mut DeviceSlice<FloatX>, mean: &mut DeviceSlice<f32>, rstd: &mut DeviceSlice<f32>,
                            inp1: &DeviceSlice<FloatX>, inp2: &DeviceSlice<FloatX>, weight: &DeviceSlice<FloatX>, bias: &DeviceSlice<FloatX>,
                            N: usize, C: usize, stream: &Stream) {
    unsafe {
        fused_residual_forward5(residual.as_raw_ptr() as *mut FloatX, normed.as_raw_ptr() as *mut FloatX, mean.as_raw_ptr() as *mut f32, rstd.as_raw_ptr() as *mut f32,
            inp1.as_raw_ptr() as *const FloatX, inp2.as_raw_ptr() as *const FloatX, weight.as_raw_ptr() as *const FloatX, bias.as_raw_ptr() as *const FloatX,
            N as i32, C as i32, stream.as_inner() as CUstream);
    }
}

#[allow(non_snake_case)]
pub fn k_layernorm_backward(dinp: &mut DeviceSlice<FloatX>, dweight: &mut DeviceSlice<FloatX>, dbias: &mut DeviceSlice<FloatX>, scratch: &mut DeviceSlice<f32>,
                            dout: &DeviceSlice<FloatX>, inp: &DeviceSlice<FloatX>, weight: &DeviceSlice<FloatX>, mean: &DeviceSlice<f32>, rstd: &DeviceSlice<f32>,
                            B: usize, T: usize, C: usize, stream: &Stream) {
    unsafe {
        layernorm_backward(dinp.as_raw_ptr() as *mut FloatX, dweight.as_raw_ptr() as *mut FloatX, dbias.as_raw_ptr() as *mut FloatX, scratch.as_raw_ptr() as *mut f32,
            dout.as_raw_ptr() as *const FloatX, inp.as_raw_ptr() as *const FloatX, weight.as_raw_ptr() as *const FloatX, mean.as_raw_ptr() as *const f32, rstd.as_raw_ptr() as *const f32,
            B as i32, T as i32, C as i32, stream.as_inner() as CUstream);
    }
}


#[allow(non_snake_case)]
pub fn k_matmul_forward_cublaslt(out: &mut DeviceSlice<FloatX>, inp: &DeviceSlice<FloatX>, weight: &DeviceSlice<FloatX>, bias: &DevicePointer<FloatX>,
                            B: usize, T: usize, C: usize, OC: usize, stream: &Stream,
                            pre_gelu: &mut DevicePointer<FloatX>, gelu_fusion: i32) {
    unsafe {
        matmul_forward_cublaslt(out.as_raw_ptr() as *mut FloatX, inp.as_raw_ptr() as *const FloatX, weight.as_raw_ptr() as *const FloatX, bias.as_ptr() as *const FloatX, B as i32, T as i32, C as i32, OC as i32, stream.as_inner() as CUstream, pre_gelu.as_raw() as *mut FloatX, gelu_fusion);
    }
}

#[allow(non_snake_case)]
pub fn k_matmul_backward(dinp: &mut DeviceSlice<FloatX>, dweight: &mut DeviceSlice<FloatX>, dbias: &mut Option<DeviceSlice<FloatX>>,
                            dout: & DeviceSlice<FloatX>, inp: & DeviceSlice<FloatX>, weight: &DeviceSlice<FloatX>, dbias_buffer: Option<DeviceSlice<f32>>,
                            B: usize, T: usize, C: usize, OC: usize, stream: &Stream,
                            pre_gelu: &mut Option<DeviceSlice<FloatX>>, gelu_fusion: i32) {
    unsafe {
        let dbias = if let Some(bias) = dbias { bias.as_raw_ptr() as *mut FloatX } else { std::ptr::null_mut() };
        let dbias_buffer = if let Some(dbias_buffer) = dbias_buffer { dbias_buffer.as_raw_ptr() as *mut f32 } else { std::ptr::null_mut() };
        let pre_gelu = if let Some(pre_gelu) = pre_gelu { pre_gelu.as_raw_ptr() as *mut FloatX } else { std::ptr::null_mut() };
        matmul_backward(dinp.as_raw_ptr() as *mut FloatX, dweight.as_raw_ptr() as *mut FloatX, dbias as *mut FloatX,
         dout.as_raw_ptr() as *const FloatX, inp.as_raw_ptr() as *const FloatX, weight.as_raw_ptr() as *const FloatX, dbias_buffer as *mut f32,
          B as i32, T as i32, C as i32, OC as i32, stream.as_inner() as CUstream, 
          pre_gelu as *mut FloatX, gelu_fusion);
    }
}

#[allow(non_snake_case)]
pub fn k_attention_forward(out: &mut DeviceSlice<FloatX>, qkvr: &mut DeviceSlice<FloatX>, att: &mut DeviceSlice<FloatX>, inp: &DeviceSlice<FloatX>, B: usize, T: usize, C: usize, NH: usize, stream: &Stream) {
    unsafe {
        attention_forward(out.as_raw_ptr() as *mut FloatX, qkvr.as_raw_ptr() as *mut FloatX, att.as_raw_ptr() as *mut FloatX, inp.as_raw_ptr() as *const FloatX, B as i32, T as i32, C as i32, NH as i32, stream.as_inner() as CUstream);
    }
}

#[allow(non_snake_case)]
pub fn k_attention_backward(dinp: &mut DeviceSlice<FloatX>, dqkvr: &mut DeviceSlice<FloatX>, datt: &mut DeviceSlice<FloatX>, scratch: &mut DeviceSlice<FloatX>,
                            dout: &DeviceSlice<FloatX>, qkvr: &DeviceSlice<FloatX>, att: &DeviceSlice<FloatX>, B: usize, T: usize, C: usize, NH: usize, stream: &Stream) {
    unsafe {
        attention_backward(dinp.as_raw_ptr() as *mut FloatX, dqkvr.as_raw_ptr() as *mut FloatX, datt.as_raw_ptr() as *mut FloatX, scratch.as_raw_ptr() as *mut FloatX,
                            dout.as_raw_ptr() as *const FloatX, qkvr.as_raw_ptr() as *const FloatX, att.as_raw_ptr() as *const FloatX, B as i32, T as i32, C as i32, NH as i32, stream.as_inner() as CUstream);
    }
}

#[allow(non_snake_case)]
pub fn k_global_sum_deterministic_float(result: &mut DeviceVariable<f32>, values: &DeviceSlice<f32>, count: usize, stream: &Stream) {
    unsafe {
        global_sum_deterministic_float(result.as_raw_ptr() as *mut f32, values.as_raw_ptr() as *const f32, count as i32, stream.as_inner() as CUstream);
    }
}

#[allow(non_snake_case)]
pub fn k_gelu_forward(out: &mut DeviceSlice<FloatX>, inp: &DeviceSlice<FloatX>, N: usize, stream: &Stream) {
    unsafe {
        gelu_forward(out.as_raw_ptr() as *mut FloatX, inp.as_raw_ptr() as *const FloatX, N as i32, stream.as_inner() as CUstream);
    }
}

pub fn k_global_norm_squared(result: DevicePointer<f32>, values: DevicePointer<FloatX>, count: usize, stride: isize, num_slices: i32, max_num_block_sums: i32, reset: bool, stream: &Stream) {
    unsafe {
        #[cfg(feature = "bf16")] global_norm_squared_bf16(result.as_raw(), values.as_raw(), count, stride, num_slices, max_num_block_sums, reset, stream.as_inner() as CUstream);
        #[cfg(feature = "fp16")] global_norm_squared_fp16(result.as_raw(), values.as_raw(), count, stride, num_slices, max_num_block_sums, reset, stream.as_inner() as CUstream);
        #[cfg(all(not(feature = "bf16"), not(feature = "fp16")))] global_norm_squared_float(result.as_raw(), values.as_raw(), count, stride, num_slices, max_num_block_sums, reset, stream.as_inner() as CUstream);
    }
}


pub fn borrow_as_f32(slice: &DeviceSlice<FloatX>) -> DeviceSlice<f32> {
    unsafe {DeviceSlice::from_raw_parts(slice.as_device_ptr().cast::<f32>(), slice.len() * size_of::<FloatX>() / size_of::<f32>())}
}

pub fn borrow_as_f32_mut(slice: &mut DeviceSlice<FloatX>) -> DeviceSlice<f32> {
    unsafe {DeviceSlice::from_raw_parts(slice.as_device_ptr().cast::<f32>(), slice.len() * size_of::<FloatX>() / size_of::<f32>())}
}

unsafe extern "C" {
    pub fn cublas_init();

    fn copy_and_cast(dst: *mut f32, src: *const FloatX, n: usize, stride_dst: usize, stride_src: usize, grid_size: usize, num_layers: usize, stream: CUstream);
    fn adamw_update(params_memory: *mut FloatX, master_params_memory: *const f32, grads_memory: *const FloatX, m_memory: *const f32, v_memory: *const f32, num_parameters: usize, w_stride: usize, g_stride: usize, s_stride: usize, num_slices: usize, learning_rate: f32, beta1: f32, beta2: f32, t: i32, eps: f32, weight_decay: f32, grad_scale: f32, seed: u32, stream: CUstream);
    fn init_from_master(params_memory: *mut FloatX, master_params_memory: *const f32, num_parameters: usize, w_stride: usize, s_stride: usize, num_slices: usize, seed: u32, stream: CUstream);

    fn fused_classifier(logits: *mut FloatX, losses: *mut f32, dloss: f32, targets: *const i32, B: i32, T: i32, V: i32, P: i32, write_dlogits: bool, stream: *mut c_void);

    fn encoder_forward(out: *mut FloatX, inp: *const i32, wte: *const FloatX, wpe: *const FloatX, B: i32, T: i32, C: i32, stream: CUstream);
    fn encoder_backward(dwte: *mut FloatX, dwpe: *mut FloatX, scratch: *mut FloatX,  // gpu outputs & scratch
        workload_indices: *mut c_int, bucket_info: *mut Int4, // cpu scratch buffers
        dout: *const FloatX, inp: *const c_int, inputs_cpu: *const c_int, // cpu/gpu inputs
        B: c_int, T: c_int, C: c_int, seed: u32, stream: CUstream);

    fn layernorm_forward(out: *mut FloatX, mean: *mut f32, rstd: *mut f32,
         inp: *const FloatX, weight: *const FloatX, bias: *const FloatX, 
         B: i32, T: i32, C: i32, stream: CUstream);
    fn residual_forward(out:  *mut FloatX, inp1: *const FloatX, inp2: *const FloatX, N: i32, stream: CUstream);
    fn fused_residual_forward5(residual: *mut FloatX, normed:  *mut FloatX, mean:  *mut f32, rstd:  *mut f32, inp1: *const FloatX, inp2: *const FloatX, weight: *const FloatX, bias: *const FloatX, N: i32, C: i32, stream: CUstream);
    fn layernorm_backward(dinp: *mut FloatX, dweight: *mut FloatX, dbias: *mut FloatX, scratch: *mut f32, dout: *const FloatX, inp: *const FloatX, weight: *const FloatX, mean: *const f32, rstd: *const f32, B: i32, T: i32, C: i32, stream: CUstream);
    
    // Matrix multiplication functions
    fn matmul_forward_cublaslt(out: *mut FloatX,
         inp: *const FloatX, weight: *const FloatX, bias: *const FloatX,
          B: i32, T: i32, C: i32, OC: i32, stream: CUstream,
           pre_gelu: *mut FloatX, gelu_fusion: i32);
    fn matmul_backward(dinp: *mut FloatX, dweight: *mut FloatX, dbias: *mut FloatX,
             dout: *const FloatX, inp: *const FloatX, weight: *const FloatX, dbias_buffer: *mut f32,
             B: i32, T: i32, C: i32, OC: i32, stream: CUstream,
             pre_gelu: *mut FloatX, gelu_fusion: i32);
    
    // Attention functions
    fn attention_forward(out: *mut FloatX, qkvr: *mut FloatX, att: *mut FloatX, inp: *const FloatX, B: i32, T: i32, C: i32, NH: i32, stream: CUstream);
    fn attention_backward(dinp: *mut FloatX, dqkvr: *mut FloatX, datt: *mut FloatX, scratch: *mut FloatX,
         dout: *const FloatX, qkvr: *const FloatX, att: *const FloatX, B: i32, T: i32, C: i32, NH: i32, stream: CUstream);

    // CUDA utils
    fn global_sum_deterministic_float(result: *mut f32, values: *const f32, count: i32, stream: CUstream);
    // fn global_sum_deterministic_fp16(result: CUdeviceptr, values: CUdeviceptr, count: i32, stream: CUstream);
    // fn global_sum_deterministic_bf16(result: CUdeviceptr, values: CUdeviceptr, count: i32, stream: CUstream);
    
    // Global norm functions
    pub fn get_max_num_block_sums(num_slices_all: *const i32, numel: i32) -> i32;
    #[cfg(all(not(feature = "bf16"), not(feature = "fp16")))] 
    fn global_norm_squared_float(out: CUdeviceptr, values: CUdeviceptr, count: usize, stride: isize, num_slices: i32, max_num_block_sums: i32, reset: bool, stream: CUstream);
    #[cfg(feature = "fp16")]
    fn global_norm_squared_fp16(out: CUdeviceptr, values: CUdeviceptr, count: usize, stride: isize, num_slices: i32, max_num_block_sums: i32, reset: bool, stream: CUstream);
    #[cfg(feature = "bf16")]
    fn global_norm_squared_bf16(out: CUdeviceptr, values: CUdeviceptr, count: usize, stride: isize, num_slices: i32, max_num_block_sums: i32, reset: bool, stream: CUstream);

    // GELU functions
    fn gelu_forward(out: *mut FloatX, inp: *const FloatX, N: i32, stream: CUstream);
    // fn gelu_backward_inplace(d_in_out: CUdeviceptr, inp: CUdeviceptr, N: i32, stream: CUstream);
}