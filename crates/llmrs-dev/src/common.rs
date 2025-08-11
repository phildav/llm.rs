use rand::{Rng, SeedableRng};
use cust::prelude::*;
use cust::memory::DeviceCopy;
use bytemuck::Zeroable;

// Newtype wrappers to implement required traits
#[cfg(feature = "bf16")]
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct Bf16(pub half::bf16);


#[cfg(feature = "bf16")]
unsafe impl DeviceCopy for Bf16 {}
#[cfg(feature = "bf16")]
unsafe impl Zeroable for Bf16 {}

#[cfg(feature = "fp16")]
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct F16(pub half::f16);

#[cfg(feature = "fp16")]
unsafe impl DeviceCopy for F16 {}
#[cfg(feature = "fp16")]
unsafe impl Zeroable for F16 {}

// ----- pick FloatX to match the .cu build flags -----
#[cfg(feature = "bf16")] pub type FloatX = Bf16;
#[cfg(feature = "fp16")] pub type FloatX = F16;
#[cfg(all(not(feature = "bf16"), not(feature = "fp16")))] pub type FloatX = f32;

pub fn make_random_i32(n: usize, max_exclusive: i32) -> Vec<i32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    (0..n).map(|_| rng.random_range(0..max_exclusive)).collect()
}

pub fn make_random_f32(n: usize) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1337);
    (0..n).map(|_| rng.random_range(-1.0f32..1.0)).collect()
}

pub fn approx_eq(a: &[f32], b: &[f32], atol: f32, rtol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x,y)| (x-y).abs() <= atol + rtol*y.abs())
}

pub fn f32_to_floatx(v: &[f32]) -> Vec<FloatX> {
    #[cfg(feature = "bf16")] { v.iter().copied().map(|x| Bf16(half::bf16::from_f32(x))).collect() }
    #[cfg(feature = "fp16")] { v.iter().copied().map(|x| F16(half::f16::from_f32(x))).collect() }
    #[cfg(all(not(feature="bf16"), not(feature="fp16")))] { v.to_vec() }
}

pub fn floatx_to_f32(v: &[FloatX]) -> Vec<f32> {
    #[cfg(feature = "bf16")] { v.iter().map(|x| x.0.to_f32()).collect() }
    #[cfg(feature = "fp16")] { v.iter().map(|x| x.0.to_f32()).collect() }
    #[cfg(all(not(feature="bf16"), not(feature="fp16")))] { v.to_vec() }
}

pub fn zero_floatx() -> FloatX {
    #[cfg(feature = "bf16")]
    { Bf16(half::bf16::from_f32(0.0)) }
    #[cfg(feature = "fp16")]
    { F16(half::f16::from_f32(0.0)) }
    #[cfg(all(not(feature="bf16"), not(feature="fp16")))]
    { 0.0f32 }
}

pub fn validate_result(device_result: &DeviceBuffer<FloatX>, cpu_reference: &[f32], name: &str, num_elements: usize, tolerance: f32) {
    let mut out_gpu = vec![zero_floatx(); num_elements];
    cust::context::CurrentContext::synchronize().unwrap();
    device_result.copy_to(&mut out_gpu).unwrap();
    let out_gpu_f32 = floatx_to_f32(&out_gpu);

    // Set epsilon based on feature flags, similar to C++ version
    #[cfg(feature = "bf16")]
    let epsilon = 0.079f32;
    #[cfg(not(feature = "bf16"))]
    let epsilon = f32::EPSILON;

    // Use iterator to find mismatches with early termination
    let faults: Vec<_> = cpu_reference
        .iter()
        .zip(out_gpu_f32.iter())
        .enumerate()
        .filter_map(|(i, (&cpu_val, &gpu_val))| {
            // Skip masked elements (non-finite values)
            if !cpu_val.is_finite() {
                return None;
            }

            // Print the first few comparisons
            if i < 5 {
                println!("{} {}", cpu_val, gpu_val);
            }

            // Effective tolerance is based on expected rounding error (epsilon),
            // plus any specified additional tolerance
            let t_eff = tolerance + cpu_val.abs() * epsilon;
            
            // Check if values differ beyond tolerance
            if (cpu_val - gpu_val).abs() > t_eff {
                Some((i, cpu_val, gpu_val))
            } else {
                None
            }
        })
        .take(10) // Limit to first 10 mismatches
        .collect();

    // Report mismatches and exit if any found
    if !faults.is_empty() {
        for (i, cpu_val, gpu_val) in faults {
            println!("Mismatch of {} at {}: CPU_ref: {} vs GPU: {}", name, i, cpu_val, gpu_val);
        }
        panic!("Validation failed");
    }
}

/// Benchmark a CUDA kernel with L2 cache clearing between runs
/// 
/// Between each run, it clears the L2 cache to ensure consistent timing measurements.
pub fn benchmark_kernel<F>(repeats: i32, kernel: F) -> Result<f32, cust::error::CudaError>
where
    F: Fn() -> Result<(), cust::error::CudaError>,
{
    use cust::event::{Event, EventFlags};
    use cust::memory::DeviceBuffer;
    use cust::stream::Stream;
    
    // Get current device and create a stream for operations
    let _device = cust::device::Device::get_device(0)?;
    let stream = Stream::new(cust::stream::StreamFlags::DEFAULT, None)?;
    
    // For L2 cache size, we'll use a reasonable default since cust doesn't expose l2_cache_size
    // Most modern GPUs have 6-40MB L2 cache, so we'll use 32MB as a safe default
    let l2_cache_size = 32 * 1024 * 1024; // 32MB
    let mut flush_buffer = DeviceBuffer::<u8>::zeroed(l2_cache_size)?;
    
    // Create CUDA events for timing
    let start_event = Event::new(EventFlags::DEFAULT)?;
    let stop_event = Event::new(EventFlags::DEFAULT)?;
    
    let mut elapsed_time = 0.0f32;
    
    for _ in 0..repeats {
        // Clear L2 cache by memsetting the flush buffer
        // We'll use a simple memset operation to clear the buffer
        let zero_buffer = vec![0u8; l2_cache_size];
        flush_buffer.copy_from(&zero_buffer)?;
        
        // Record start time
        start_event.record(&stream)?;
        
        // Launch the kernel
        kernel()?;
        
        // Record stop time
        stop_event.record(&stream)?;
        
        // Synchronize events
        start_event.synchronize()?;
        stop_event.synchronize()?;
        
        // Calculate elapsed time for this run
        let single_call = stop_event.elapsed_time_f32(&start_event)?;
        elapsed_time += single_call;
    }
    
    // Return average time
    Ok(elapsed_time / repeats as f32)
}


