// Declaration of CUDA launcher we are using in the Rust code.
use cust::sys::CUstream;
use cust::sys::CUdeviceptr;


unsafe extern "C" {
    pub fn copy_and_cast(dst: CUdeviceptr, src: CUdeviceptr, n: usize, stride_dst: usize, stride_src: usize, grid_size: usize, num_layers: usize, stream: CUstream);
    pub fn adamw_update(params_memory: CUdeviceptr, master_params_memory: CUdeviceptr, grads_memory: CUdeviceptr, m_memory: CUdeviceptr, v_memory: CUdeviceptr, num_parameters: usize, w_stride: usize, g_stride: usize, s_stride: usize, num_slices: usize, learning_rate: f32, beta1: f32, beta2: f32, t: i32, eps: f32, weight_decay: f32, grad_scale: f32, seed: u32, stream: CUstream);
    pub fn init_from_master(params_memory: CUdeviceptr, master_params_memory: CUdeviceptr, num_parameters: usize, w_stride: usize, s_stride: usize, num_slices: usize, seed: u32, stream: CUstream);

    pub fn fused_classifier(logits: CUdeviceptr, losses: CUdeviceptr, dloss: f32, targets: CUdeviceptr, B: i32, T: i32, V: i32, P: i32, write_dlogits: bool, stream: CUstream);

    pub fn encoder_forward(out: CUdeviceptr, inp: CUdeviceptr, wte: CUdeviceptr, wpe: CUdeviceptr, B: i32, T: i32, C: i32, stream: CUstream);
    pub fn encoder_backward(dwte: CUdeviceptr, dwpe: CUdeviceptr, scratch: CUdeviceptr, workload_indices: CUdeviceptr, bucket_info: CUdeviceptr, dout: CUdeviceptr, inp: CUdeviceptr, inputs_cpu: CUdeviceptr, B: i32, T: i32, C: i32, seed: u32, stream: CUstream);

    pub fn layernorm_forward(out: CUdeviceptr, mean: CUdeviceptr, rstd: CUdeviceptr, inp: CUdeviceptr, weight: CUdeviceptr, bias: CUdeviceptr, B: i32, T: i32, C: i32, stream: CUstream);
    pub fn residual_forward(out: CUdeviceptr, inp1: CUdeviceptr, inp2: CUdeviceptr, N: i32, stream: CUstream);
    pub fn fused_residual_forward5(residual: CUdeviceptr, normed: CUdeviceptr, mean: CUdeviceptr, rstd: CUdeviceptr, inp1: CUdeviceptr, inp2: CUdeviceptr, weight: CUdeviceptr, bias: CUdeviceptr, N: i32, C: i32, stream: CUstream);
    pub fn layernorm_backward(dinp: CUdeviceptr, dweight: CUdeviceptr, dbias: CUdeviceptr, scratch: CUdeviceptr, dout: CUdeviceptr, inp: CUdeviceptr, weight: CUdeviceptr, mean: CUdeviceptr, rstd: CUdeviceptr, B: i32, T: i32, C: i32, stream: CUstream);
}