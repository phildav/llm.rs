use std::f32::consts::PI;

/// Learning rate scheduler that supports various scheduling strategies
#[derive(Debug, Clone)]
pub struct LearningRateScheduler {
    /// Type of scheduler: "cosine", "linear", "constant", "wsd"
    pub scheduler_type: String,
    pub learning_rate: f32,
    pub warmup_iterations: i32,
    pub train_num_batches: i32,
    pub final_learning_rate_frac: f32,
}


impl LearningRateScheduler {
    /// Create a new learning rate scheduler
    pub fn new(
        scheduler_type: &str,
        learning_rate: f32,
        warmup_iterations: i32,
        train_num_batches: i32,
        final_learning_rate_frac: f32,
    ) -> Self {
        Self {
            scheduler_type: scheduler_type.to_string(),
            learning_rate,
            warmup_iterations,
            train_num_batches,
            final_learning_rate_frac,
        }
    }

    /// cosine: warmup linearly to max LR, then cosine decay to LR * final_learning_rate_frac
    pub fn get_learning_rate_cosine(&self, step: i32) -> f32 {
        let lr;
        if step < self.warmup_iterations {
            lr = self.learning_rate * ((step + 1) as f32) / self.warmup_iterations as f32;
        } else {
            let decay_ratio = ((step - self.warmup_iterations) as f32) 
                / (self.train_num_batches - self.warmup_iterations) as f32;
            assert!(0.0 <= decay_ratio && decay_ratio <= 1.0);
            let coeff = 0.5 * (1.0 + (PI * decay_ratio).cos()); // coeff starts at 1 and goes to 0
            assert!(0.0 <= coeff && coeff <= 1.0);
            let min_lr = self.learning_rate * self.final_learning_rate_frac;
            lr = min_lr + coeff * (self.learning_rate - min_lr);
        }
        lr
    }

    /// linear: warmup linearly to max LR, then decay linearly to LR * final_learning_rate_frac
    pub fn get_learning_rate_linear(&self, step: i32) -> f32 {
        let lr ;
        if step < self.warmup_iterations {
            lr = self.learning_rate * ((step + 1) as f32) / self.warmup_iterations as f32;
        } else {
            let decay_ratio = ((step - self.warmup_iterations) as f32) 
                / (self.train_num_batches - self.warmup_iterations) as f32;
            assert!(0.0 <= decay_ratio && decay_ratio <= 1.0);
            let min_lr = self.learning_rate * self.final_learning_rate_frac;
            lr = self.learning_rate - decay_ratio * (self.learning_rate - min_lr);
        }
        lr
    }

    /// wsd schedule: warmup linearly, keep constant, last 20% decay using 1 - sqrt decay to final_frac (should be 0.0)
    /// https://arxiv.org/abs/2405.18392
    pub fn get_learning_rate_constant(&self, _step: i32) -> f32 {
        self.learning_rate
    }

    /// Get the learning rate for a given step using WSD scheduling
    /// WSD: warmup linearly, keep constant, last 20% decay using 1 - sqrt decay
    pub fn get_learning_rate_wsd(&self, step: i32) -> f32 {
        let decay_point = (0.8 * self.train_num_batches as f32) as i32;
        let max_lr = self.learning_rate;
        let mut lr = max_lr;
        
        if step < self.warmup_iterations {
            let decay_ratio = ((step + 1) as f32) / self.warmup_iterations as f32;
            lr = max_lr * decay_ratio;
        } else if step < decay_point {
            // noop, keep lr constant
        } else {
            let decay_ratio = ((step - decay_point) as f32) / (self.train_num_batches - decay_point) as f32;
            debug_assert!(0.0 <= decay_ratio && decay_ratio <= 1.0);
            let min_lr = max_lr * self.final_learning_rate_frac;
            return min_lr + (1.0 - decay_ratio.sqrt()) * (max_lr - min_lr);
        }
        lr
    }

    /// return the learning rate at a given step
    pub fn get_learning_rate(&self, step: i32) -> f32 {
        match self.scheduler_type.as_str() {
            "cosine" => self.get_learning_rate_cosine(step),
            "linear" => self.get_learning_rate_linear(step),
            "constant" => self.get_learning_rate_constant(step),
            "wsd" => self.get_learning_rate_wsd(step),
            _ => {
                eprintln!("Unknown learning rate scheduler type: {}", self.scheduler_type);
                self.learning_rate // fallback to base learning rate
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to compare floating-point values with a hardcoded tolerance
    fn assert_approx_eq(actual: f32, expected: f32) {
        const TOLERANCE: f32 = 1e-6;
        assert!(
            (actual - expected).abs() < TOLERANCE,
            "Expected {} to be approximately equal to {} (tolerance: {})",
            actual,
            expected,
            TOLERANCE
        );
    }

    #[test]
    fn test_constant_scheduler() {
        let scheduler = LearningRateScheduler::new("constant", 0.001, 0, 100, 1.0);
        assert_eq!(scheduler.get_learning_rate(0), 0.001);
        assert_eq!(scheduler.get_learning_rate(50), 0.001);
        assert_eq!(scheduler.get_learning_rate(100), 0.001);
    }

    #[test]
    fn test_cosine_scheduler() {
        let scheduler = LearningRateScheduler::new("cosine", 0.001, 10, 100, 0.1);
        // During warmup
        assert_approx_eq(scheduler.get_learning_rate(0), 0.0001);
        assert_approx_eq(scheduler.get_learning_rate(5), 0.0006);
        assert_approx_eq(scheduler.get_learning_rate(10), 0.001);
        // After warmup, should decay
        let lr_50 = scheduler.get_learning_rate(50);
        assert!(lr_50 < 0.001 && lr_50 > 0.0001);
    }

    #[test]
    fn test_linear_scheduler() {
        let scheduler = LearningRateScheduler::new("linear", 0.001, 10, 100, 0.1);
        // During warmup
        assert_approx_eq(scheduler.get_learning_rate(0), 0.0001);
        assert_approx_eq(scheduler.get_learning_rate(5), 0.0006);
        assert_approx_eq(scheduler.get_learning_rate(10), 0.001);
        // After warmup, should decay linearly
        let lr_50 = scheduler.get_learning_rate(50);
        assert!(lr_50 < 0.001 && lr_50 > 0.0001);
    }
}
