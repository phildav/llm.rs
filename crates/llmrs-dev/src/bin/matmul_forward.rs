use cust::prelude::*;
use cust::error::CudaResult;
use llmrs_dev::common::*;

fn matmul_forward_cpu_rs(out: &mut [f32], inp: &[f32], weight: &[f32], bias: &[f32], B: usize, T: usize, C: usize, OC: usize) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    for b in 0..B {
        for t in 0..T {
            let out_bt = &mut out[(b*T + t)*OC .. (b*T + t + 1)*OC];
            let inp_bt = &inp[(b*T + t)*C .. (b*T + t + 1)*C];
            for o in 0..OC {
                let mut val = bias[o];
                let wrow = &weight[o*C .. (o+1)*C];
                for i in 0..C {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

// C-ABI dispatcher exported by the .cu (linked by build.rs)
unsafe extern "C" {
    fn matmul_forward_cpu(
        out: *mut f32,
        inp: *const f32,
        weight: *const f32,
        bias: *const f32,
        B: i32, T: i32, C: i32, OC: i32,
    );

    fn matmul_forward(
        kernel_num: i32,
        out: *mut f32,
        inp: *const f32,
        weight: *const f32,
        bias: *const f32,
        B: i32, T: i32, C: i32, OC: i32,
        sqrt_block_size: i32,
    );

    fn cublas_init();
}

#[allow(non_snake_case)]
fn main() -> CudaResult<()> {
    let B: usize = 32;
    let T: usize = 1024;
    let C: usize = 768;
    let OC: usize = 768 * 4; // expansion of 4, e.g. in the MLP

    // CUDA init (cust)
    let _ctx = cust::quick_init()?;
    let _stream = Stream::new(StreamFlags::DEFAULT, None)?;
    unsafe { cublas_init() };

    // create host memory of random numbers
    let mut out_cpu = vec![0f32; B*T*OC];
    let inp = make_random_f32(B*T*C);
    let weight = make_random_f32(OC*C);
    let bias = make_random_f32(OC);

    // move to GPU 
    let d_out: DeviceBuffer<f32> = DeviceBuffer::zeroed(B*T*OC)?;
    let d_inp: DeviceBuffer<f32> = DeviceBuffer::from_slice(&inp)?;
    let d_weight: DeviceBuffer<f32> = DeviceBuffer::from_slice(&weight)?;
    let d_bias: DeviceBuffer<f32> = DeviceBuffer::from_slice(&bias)?;
    
    let kernel_num: i32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    println!("Using kernel {}", kernel_num);

    // first check the correctness of the kernel
    matmul_forward_cpu_rs(&mut out_cpu, &inp, &weight, &bias, B, T, C, OC);

    // time the kernel at different block sizes
    let sqrt_block_sizes = [4, 8, 16, 32];
    let tol = 1e-1f32;

    for &sqrt_block_size in &sqrt_block_sizes {
        println!("Checking block size {} x {}", sqrt_block_size, sqrt_block_size);
        unsafe {
            matmul_forward(
                kernel_num,
                d_out.as_device_ptr().as_raw() as *mut f32,
                d_inp.as_device_ptr().as_ptr(),
                d_weight.as_device_ptr().as_ptr(),
                d_bias.as_device_ptr().as_ptr(),
                B as i32, T as i32, C as i32, OC as i32,
                sqrt_block_size as i32,
            );
        }
        
        validate_result_f32(&d_out, &out_cpu, "out", B*T*OC, tol);
    }
    
    println!("All results match. Starting benchmarks.\n");

    // Benchmark with CUDA Events (cust)
    for &sqrt_block_size in &sqrt_block_sizes {
        let repeat = 100;
        
        // Measure total wall time
        let elapsed_time = benchmark_kernel(repeat, || {
            unsafe {
                matmul_forward(
                    kernel_num,
                    d_out.as_device_ptr().as_raw() as *mut f32,
                    d_inp.as_device_ptr().as_raw() as *const f32,
                    d_weight.as_device_ptr().as_raw() as *const f32,
                    d_bias.as_device_ptr().as_raw() as *const f32,
                    B as i32, T as i32, C as i32, OC as i32,
                    sqrt_block_size as i32,
                );
            }
            Ok(())
        })?;

        // napkin math: estimate the flops achieved
        // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
        let flops = B * T * C * OC * 2; // 2 flops per multiply-add
        let tflops = flops as f32 / elapsed_time / 1e9;
        println!("sqrt_block_size {:4} | time {:8.4} ms | tflops {:6.2}", sqrt_block_size, elapsed_time, tflops);
    }

    Ok(())
}
