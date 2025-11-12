use std::{
    fs::{File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
};

pub struct Logger {
    active: bool,
    output_log_file: PathBuf,
}

impl Logger {
    pub fn init(log_dir: Option<&str>, process_rank: i32, resume: bool) -> io::Result<Self> {
        let mut logger = Logger { active: false, output_log_file: PathBuf::new() };

        if let Some(dir) = log_dir && process_rank == 0 {
            logger.active = true;
            logger.output_log_file = Path::new(dir).join("main.log");

            if !resume {
                // wipe
                File::create(&logger.output_log_file)?; // truncate or create
            }
        }
        Ok(logger)
    }

    #[inline]
    pub fn log_eval(&self, step: i32, val: f32) -> io::Result<()> {
        if !self.active { return Ok(()); }
        let mut f = OpenOptions::new().create(true).append(true).open(&self.output_log_file)?;
        writeln!(f, "s:{} eval:{:.4}", step, val)
    }

    #[inline]
    pub fn log_val(&self, step: i32, val_loss: f32) -> io::Result<()> {
        if !self.active { return Ok(()); }
        let mut f = OpenOptions::new().create(true).append(true).open(&self.output_log_file)?;
        writeln!(f, "s:{} tel:{:.4}", step, val_loss)
    }

    #[inline]
    pub fn log_train(&self, step: i32, train_loss: f32, learning_rate: f32, grad_norm: f32) -> io::Result<()> {
        if !self.active { return Ok(()); }
        let mut f = OpenOptions::new().create(true).append(true).open(&self.output_log_file)?;
        writeln!(f, "s:{} trl:{:.4} lr:{:.6} norm:{:.2}", step, train_loss, learning_rate, grad_norm)
    }
}