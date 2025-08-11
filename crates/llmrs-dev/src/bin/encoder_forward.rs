use cust::prelude::*;
use cust::error::CudaResult;
use llmrs_dev::common::*;

#[allow(non_snake_case)]
fn encoder_forward_cpu(out: &mut [f32], inp: &[i32], wte: &[f32], wpe: &[f32], B: usize, T: usize, C: usize) {
    for b in 0..B {
        for t in 0..T {
            let out_bt = &mut out[(b*T + t)*C .. (b*T + t + 1)*C];
            let ix = inp[b*T + t] as usize;
            let wte_ix = &wte[ix*C .. (ix+1)*C];
            let wpe_t  = &wpe[t*C  .. (t+1)*C];
            for i in 0..C { out_bt[i] = wte_ix[i] + wpe_t[i]; }
        }
    }
}


// C-ABI dispatcher exported by the .cu (linked by build.rs)
unsafe extern "C" {
    fn encoder_forward(
        kernel_num: i32,
        out: *mut FloatX,
        inp: *const i32,
        wte: *const FloatX,
        wpe: *const FloatX,
        B: i32, T: i32, C: i32,
        block_size: i32,
    );
}


#[allow(non_snake_case)]
fn main() -> CudaResult<()> {
    let B: usize = 8;
    let T: usize = 1024;
    let C: usize = 768;
    let V: usize = 50257;

    // CUDA init (cust)
    let _ctx = cust::quick_init()?;
    let _stream = Stream::new(StreamFlags::DEFAULT, None)?;

    // create host memory of random numbers
    let mut out_cpu = vec![0f32; B*T*C];
    let inp = make_random_i32(B*T, V as i32);
    let wte_f32 = make_random_f32(V*C);
    let wpe_f32 = make_random_f32(T*C);

    // move to GPU 
    let d_out: DeviceBuffer<FloatX> = DeviceBuffer::zeroed(B*T*C)?;
    let d_inp: DeviceBuffer<i32> = DeviceBuffer::from_slice(&inp)?;
    // Convert to FloatX for device
    let wte_x = f32_to_floatx(&wte_f32);
    let wpe_x = f32_to_floatx(&wpe_f32);
    let d_wte: DeviceBuffer<FloatX> = DeviceBuffer::from_slice(&wte_x)?;
    let d_wpe: DeviceBuffer<FloatX> = DeviceBuffer::from_slice(&wpe_x)?;
    
    let kernel_num: i32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    println!("Using kernel {}", kernel_num);

    // first check the correctness of the kernel
    encoder_forward_cpu(&mut out_cpu, &inp, &wte_f32, &wpe_f32, B, T, C);

    // time the kernel at different block sizes
    let block_sizes = [32, 64, 128, 256, 512, 1024];
    let tol = if cfg!(any(feature="bf16", feature="fp16")) { 1e-2 } else { 1e-5 };

    let _out_host_x = vec![zero_floatx(); B*T*C];
    for &block_size in &block_sizes {
        println!("Checking block size {block_size}");
        unsafe {
            encoder_forward(
                kernel_num,
                d_out.as_device_ptr().as_raw() as *mut FloatX,
                d_inp.as_device_ptr().as_ptr(),
                d_wte.as_device_ptr().as_ptr(),
                d_wpe.as_device_ptr().as_ptr(),
                B as i32, T as i32, C as i32, block_size as i32,
            );
        }
        
        validate_result(&d_out, &out_cpu, "out", B*T*C, tol);
    }
    
    println!("All results match. Starting benchmarks.\n");

    // Benchmark with CUDA Events (cust)
    for &bs in &block_sizes {
        let repeat = 1000;
        let elapsed_time = benchmark_kernel(repeat, || {
            unsafe {
                encoder_forward(
                    kernel_num,
                    d_out.as_device_ptr().as_raw() as *mut FloatX,
                    d_inp.as_device_ptr().as_raw() as *const i32,
                    d_wte.as_device_ptr().as_raw() as *const FloatX,
                    d_wpe.as_device_ptr().as_raw() as *const FloatX,
                    B as i32, T as i32, C as i32, bs as i32,
                );
            }
            Ok(())
        })?;

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 3 reads and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        let memory_ops = B * T * C * 4 * 4;
        let memory_bandwidth = memory_ops as f32 / elapsed_time / 1e6;
        
        println!("block_size {:4} | time {:8.4} ms | bandwidth {:6.2} GB/s", bs, elapsed_time, memory_bandwidth);
    }

    Ok(())
}
