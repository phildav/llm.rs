use cust::prelude::*;
use cust::error::CudaResult;

#[allow(unused_imports)]
use llmrs_dev::common as _;  // Force linking using settings from build.rs. Otherwise, the kernel will not be linked.

unsafe extern "C" {
    fn add_cuda(a: *const f32, b: *const f32, out: *mut f32, n: usize);
}

pub fn add(a: &[f32], b: &[f32]) -> CudaResult<Vec<f32>> {
    assert_eq!(a.len(), b.len(), "Input slices must have the same length");
    
    let n = a.len();

    // Initialize CUDA context
    let _ctx = cust::quick_init()?;
    
    // Create device buffers
    let d_a = DeviceBuffer::from_slice(a)?;
    let d_b = DeviceBuffer::from_slice(b)?;
    let d_result = DeviceBuffer::zeroed(n)?;

    // Launch kernel
    unsafe {
        add_cuda(
            d_a.as_device_ptr().as_ptr(),
            d_b.as_device_ptr().as_ptr(),
            d_result.as_device_ptr().as_ptr() as *mut f32,
            n,
        );
    }

    // Synchronize and copy result back
    cust::context::CurrentContext::synchronize()?;
    let mut result = vec![0.0f32; n];
    d_result.copy_to(&mut result)?;

    Ok(result)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![10f32, 20.0, 30.0, 40.0];
    let c = add(&a, &b)?;
    println!("{:?} + {:?} = {:?}", a, b, c);
    Ok(())
}
