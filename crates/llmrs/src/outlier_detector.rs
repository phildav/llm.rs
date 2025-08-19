/*
Simple OutlierDetector that we can use to monitor the loss and grad norm
Internally, it keeps track of a window of measurements and each time we
add a measurement, it returns the z-score of the new value with respect to
the window of measurements. This can be used to detect outliers in the data.

We use double so that the detector doesn't drift too much, because we
update the mean and variance with += on each step for efficiency. We could
reconsider this choice in the future, as the compute cost here is minimal.
*/

use std::f64::NAN;

// use compile-time constant for window size to avoid dynamic memory allocations
const OUTLIER_DETECTOR_WINDOW_SIZE: usize = 128;

pub struct OutlierDetector {
    buffer: [f64; OUTLIER_DETECTOR_WINDOW_SIZE],
    count: usize,
    index: usize,
    sum: f64,
    sum_sq: f64,
}

impl OutlierDetector {
    pub fn new() -> Self {
        Self {
            buffer: [0.0; OUTLIER_DETECTOR_WINDOW_SIZE],
            count: 0,
            index: 0,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    pub fn update(&mut self, new_value: f64) -> f64 {
        if self.count < OUTLIER_DETECTOR_WINDOW_SIZE {
            // here we are still building up a window of observations
            self.buffer[self.count] = new_value;
            self.sum += new_value;
            self.sum_sq += new_value * new_value;
            self.count += 1;
            NAN // not enough data yet
        } else {
            // we've filled the window, so now we can start detecting outliers

            // pop the oldest value from the window
            let old_value = self.buffer[self.index];
            self.sum -= old_value;
            self.sum_sq -= old_value * old_value;
            // push the new value into the window
            self.buffer[self.index] = new_value;
            self.sum += new_value;
            self.sum_sq += new_value * new_value;
            // move the index to the next position
            self.index = (self.index + 1) % OUTLIER_DETECTOR_WINDOW_SIZE;
            // calculate the z-score of the new value
            let mean = self.sum / OUTLIER_DETECTOR_WINDOW_SIZE as f64;
            let variance = (self.sum_sq / OUTLIER_DETECTOR_WINDOW_SIZE as f64) - (mean * mean);
            let std_dev = variance.sqrt();
            if std_dev == 0.0 {
                return 0.0;
            }
            let z = (new_value - mean) / std_dev;

            z
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /*
    Tests our OutlierDetector
    
    Run with: cargo test test_outlier_detector
    */
    
    #[test]
    fn test_outlier_detector() {
        let mut detector = OutlierDetector::new();

        // Initialize random number generator with seed 1337
        let mut rng_seed: u32 = 1337;

        // generate OUTLIER_DETECTOR_WINDOW_SIZE * 2 random numbers between -1 and 1
        for i in 0..OUTLIER_DETECTOR_WINDOW_SIZE * 2 {
            // C's rand() equivalent: linear congruential generator
            rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
            let rand_val = (rng_seed >> 16) & 0x7fff; // Extract 15 bits like C's rand()
            let val = (rand_val as f64) / 32767.0 * 2.0 - 1.0; // Random number between -1 and 1
            let zscore = detector.update(val);

            println!("Step {}: Value = {:.4}, zscore = {:.4}", i, val, zscore);

            // check that the first OUTLIER_DETECTOR_WINDOW_SIZE values return nan
            if i < OUTLIER_DETECTOR_WINDOW_SIZE {
                assert!(zscore.is_nan(), "Expected nan, got {:.4}", zscore);
            } else {
                // check that the zscore is within reasonable bounds
                assert!(
                    zscore >= -3.0 && zscore <= 3.0,
                    "Z-score {:.4} is outside of expected range",
                    zscore
                );
            }
        }

        // simulate an outlier
        let outlier = 10.0; // <--- loss spike
        let zscore = detector.update(outlier);
        println!("Outlier Step: Value = {:.4}, zscore = {:.4}", outlier, zscore);

        // check that the z-score here is large
        assert!(
            zscore >= 5.0,
            "Z-score {:.4} is not large enough for an outlier",
            zscore
        );

        println!("OK");
    }
}
