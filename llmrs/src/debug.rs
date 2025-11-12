use crate::common::{FloatX, ToF32};
use std::cmp::min;
use cust::{prelude::*};


pub fn sig_f32(x: &[f32]) -> u64 {
    let mut h = 0u64;
    for (i, &v) in x.iter().enumerate() {
        let u = v.to_bits() as u64;
        h ^= u.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(i as u64);
    }
    h
}

// if BF16 is stored as u16 on host:
pub fn sig_bf16(words: &[u16]) -> u64 {
    let mut h = 0u64;
    for (i, &w) in words.iter().enumerate() {
        h ^= (w as u64) << ((i & 3) * 16);
    }
    h
}


pub fn debug_print_tensor(cpu_tensor: &[FloatX], name: &str) {        
    // Compute checksum before f32 conversion for bf16
    let bf16_checksum = if std::any::type_name::<FloatX>() == "bf16" {
        let bf16_words: &[u16] = bytemuck::cast_slice(cpu_tensor);
        sig_bf16(bf16_words)
    } else {
        0 // Will be computed in f32 version
    };
    
    let f32_buffer: Vec<f32> = cpu_tensor.iter().map(|&x| x.to_f32()).collect();
    debug_print_tensor_f32(&f32_buffer, name, bf16_checksum);
}

pub fn debug_print_tensor_f32(cpu_tensor: &[f32], name: &str, bf16_checksum: u64) {
    let first_5 = &cpu_tensor[..min(5, cpu_tensor.len())];
    let last_5 = if cpu_tensor.len() > 5 {
        &cpu_tensor[cpu_tensor.len() - 5..]
    } else {
        &[]
    };
    let checksum_str = if bf16_checksum != 0 {
        format!(", bf16_checksum: 0x{:x}", bf16_checksum)
    } else {
        let f32_checksum = sig_f32(cpu_tensor);
        format!(", f32_checksum: 0x{:x}", f32_checksum)
    };
    println!("DEBUG TENSOR {} (sum: {}){} (first 5: {:?}, last 5: {:?})", name, cpu_tensor.iter().sum::<f32>(), checksum_str, first_5, last_5);
}

pub fn debug_print_device_tensor(tensor: DeviceSlice<FloatX>, name: &str, stream: &Stream) {
    stream.synchronize().unwrap();
    let mut cpu_buffer = vec![unsafe { std::mem::zeroed::<FloatX>() }; tensor.len()];
    tensor.copy_to(&mut cpu_buffer).unwrap();
    
    debug_print_tensor(&cpu_buffer.as_slice(), name);
}