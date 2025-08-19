/*
Implements a simple Sampler, used during model inference to sample tokens.
*/

// Simple xorshift RNG
pub fn random_u32(state: &mut u64) -> u32 {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ((*state * 0x2545F4914F6CDD1D) >> 32) as u32
}

pub fn random_f32(state: &mut u64) -> f32 { // random float32 in [0,1)
    (random_u32(state) >> 8) as f32 / 16777216.0f32
}

pub fn sample_softmax(logits: &[f32], coin: f32) -> usize {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    let norm: f64 = logits.iter().map(|&x| f64::from(x.exp())).sum();
    // instead of dividing all exp(logits), we can just multiply coin.
    let coin = coin as f64 * norm;
    let mut cdf = 0.0f64;
    for (i, &logit) in logits.iter().enumerate() {
        cdf += logit.exp() as f64;
        if coin < cdf {
            return i;
        }
    }
    logits.len() - 1 // in case of rounding errors
}

