

// Newtype wrappers to implement required traits
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct Bf16(pub half::bf16);

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct F16(pub half::f16);

use cust::memory::DeviceCopy;
use bytemuck::{Zeroable, NoUninit, AnyBitPattern};

unsafe impl DeviceCopy for Bf16 {}
unsafe impl Zeroable for Bf16 {}
unsafe impl NoUninit for Bf16 {}
unsafe impl AnyBitPattern for Bf16 {}

unsafe impl DeviceCopy for F16 {}
unsafe impl Zeroable for F16 {}
unsafe impl NoUninit for F16 {}
unsafe impl AnyBitPattern for F16 {}

/// Precision mode enum for runtime selection and configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionMode {
    Fp32,
    Fp16,
    Bf16,
}

impl PrecisionMode {
    /// Get the precision mode based on compile-time features
    pub const fn from_features() -> Self {
        #[cfg(feature = "bf16")]
        return PrecisionMode::Bf16;
        
        #[cfg(feature = "fp16")]
        return PrecisionMode::Fp16;
        
        #[cfg(all(not(feature = "bf16"), not(feature = "fp16")))]
        return PrecisionMode::Fp32;
    }
    
    /// Get the name of the precision mode as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            PrecisionMode::Fp32 => "fp32",
            PrecisionMode::Fp16 => "fp16", 
            PrecisionMode::Bf16 => "bf16",
        }
    }
    
    /// Get the size in bytes of the precision type
    pub fn size_bytes(&self) -> usize {
        match self {
            PrecisionMode::Fp32 => 4,
            PrecisionMode::Fp16 => 2,
            PrecisionMode::Bf16 => 2,
        }
    }
}

pub static PRECISION_MODE: PrecisionMode = PrecisionMode::from_features();

// ----- pick FloatX to match the .cu build flags -----
#[cfg(feature = "bf16")] pub type FloatX = Bf16;
#[cfg(feature = "fp16")] pub type FloatX = F16;
#[cfg(all(not(feature = "bf16"), not(feature = "fp16")))] pub type FloatX = f32;


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