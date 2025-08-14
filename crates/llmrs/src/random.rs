use rand_mt::Mt;
use std::f32::consts::PI;


pub struct Mt19937 {
    rng: Mt,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        Self {
            rng: Mt::new(seed),
        }
    }

    pub fn randu32(&mut self) -> u32 {
        self.rng.next_u32()
    }

    pub fn randu64(&mut self) -> u64 {
        ((self.randu32() as u64) << 32) | (self.randu32() as u64)
    }

    pub fn randf32(&mut self) -> f32 {
        ((self.randu32() as u64) & ((1u64 << 24) - 1)) as f32 * (1.0f32 / (1u64 << 24) as f32)
    }

    pub fn randf64(&mut self) -> f64 {
        (self.randu64() & ((1u64 << 53) - 1)) as f64 * (1.0f64 / (1u64 << 53) as f64)
    }

}


/// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
/// Processes 16 values at a time for efficiency
pub fn normal_fill_16(data: &mut [f32], mean: f32, std: f32) {
    const EPSILON: f32 = 1e-12f32;
    
    for t in 0..8 {
        let u1 = 1.0f32 - data[t];
        let u2 = data[t + 8];
        let radius = (-2.0f32 * (u1 + EPSILON).ln()).sqrt();
        let theta = 2.0f32 * PI * u2;
        data[t] = radius * theta.cos() * std + mean;
        data[t + 8] = radius * theta.sin() * std + mean;
    }
}

/// Fill an array with normally distributed random numbers using Box-Muller transform
pub fn normal_fill(data: &mut [f32], mean: f32, std: f32, rng: &mut Mt19937) {
    let numel = data.len();
    
    // First, fill with uniform random numbers
    for t in 0..numel {
        data[t] = rng.randf32();
    }
    
    // Process in chunks of 16
    for i in (0..numel - 15).step_by(16) {
        normal_fill_16(&mut data[i..i + 16], mean, std);
    }
    
    // Handle remaining elements if numel % 16 != 0
    if numel % 16 != 0 {
        let start = numel - 16;
        // Recompute the last 16 values
        for i in 0..16 {
            data[start + i] = rng.randf32();
        }
        normal_fill_16(&mut data[start..], mean, std);
    }
}

/// Generate normally distributed random numbers
pub fn normal_(data: &mut [f32], mean: f32, std: f32, rng: &mut Mt19937) {
    const EPSILON: f32 = 1e-12f32;
    let numel = data.len();
    
    if numel >= 16 {
        normal_fill(data, mean, std, rng);
    } else {
        // For small arrays, use double precision for better accuracy
        let mut next_double_normal_sample = 0.0f64;
        let mut has_next_double_normal_sample = false;
        
        for t in 0..numel {
            if has_next_double_normal_sample {
                data[t] = (next_double_normal_sample * std as f64 + mean as f64) as f32;
                has_next_double_normal_sample = false;
                continue;
            }
            
            // For numel < 16 we draw a double (float64)
            let u1 = rng.randf64() as f32;
            let u2 = rng.randf64() as f32;
            let radius = (-2.0f32 * (1.0f32 - u2 + EPSILON).ln()).sqrt();
            let theta = 2.0f32 * PI * u1;
            next_double_normal_sample = (radius * theta.sin()) as f64;
            has_next_double_normal_sample = true;
            data[t] = radius * theta.cos() * std + mean;
        }
    }
}


pub fn random_permutation(data: &mut [usize], numel: usize, rng: &mut Mt) {
    for i in (1..numel).rev() {
        // pick an index j in [0, i] with equal probability
        let j = rng.next_u32() as usize %  (i + 1);
        // swap i <-> j
        let tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

pub fn init_identity_permutation(data: &mut [usize]) {
    for i in 0..data.len() {
        data[i] = i;
    }
}

#[cfg(test)]
mod tests {
    use crate::random::{Mt19937, normal_};

    #[test]
    fn mt19937_numerical_id() {
        // PyTorch/llm.c compatible seed
        let seed = 137u32;
        let mut rng = Mt19937::new(seed);

        // Integer values
        let expected = [
            4053805790u32,
            2173880614u32,
            380293709u32,
            1237255315u32,
            2986595568u32,
        ];

        for &exp in &expected {
            let val = rng.randu32();
            assert_eq!(val, exp);
        }

        // Test normal distribution with t8 array
        let mut t8 = [0.0f32; 8];
        normal_(&mut t8, 0.0, 1.0, &mut rng);
        
        let expected_t8 = [
            0.7947664260864258f32,
            1.4369317293167114f32,
            -0.2292192131280899f32,
            0.47556325793266296f32,
            -0.6334410905838013f32,
            -0.5791953802108765f32,
            -0.0925704762339592f32,
            -0.8659197092056274f32,
        ];
        
        for (i, &exp) in expected_t8.iter().enumerate() {
            assert!((t8[i] - exp).abs() < 1e-6, "t8[{}]: expected {}, got {}", i, exp, t8[i]);
        }
        
        // Test next random integer
        let next_int = rng.randu32();
        assert_eq!(next_int, 2186503452u32);
        
        // Test normal distribution with t16 array
        let mut t16 = [0.0f32; 16];
        normal_(&mut t16, 0.0, 1.0, &mut rng);
        
        let expected_t16 = [
            -1.2813878059387207f32,
            -2.646395683288574f32,
            -0.06569503247737885f32,
            0.2180829495191574f32,
            -0.46536165475845337f32,
            -0.33108410239219666f32,
            2.5485482215881348f32,
            0.10425379872322083f32,
            0.8460659980773926f32,
            0.9462448358535767f32,
            -0.2913765013217926f32,
            0.34313806891441345f32,
            -1.1186704635620117f32,
            -0.18305328488349915f32,
            -2.3153159618377686f32,
            0.3961987793445587f32,
        ];
        
        for (i, &exp) in expected_t16.iter().enumerate() {
            assert!((t16[i] - exp).abs() < 1e-6, "t16[{}]: expected {}, got {}", i, exp, t16[i]);
        }
        
        // Test final random integer
        let final_int = rng.randu32();
        assert_eq!(final_int, 2756748748u32);
    }

}