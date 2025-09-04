#![allow(clippy::needless_range_loop)]

use std::io::BufRead;
use std::fs;
use std::path::Path;


pub fn read_le_u16<R: BufRead>(reader: &mut R) -> u16 {
    let mut buf = [0u8; 2];

    reader.read_exact(&mut buf).unwrap();
    u16::from_le_bytes(buf)
}

pub fn read_le_i16<R: BufRead>(reader: &mut R) -> i16 {
    let mut buf = [0u8; 2];

    reader.read_exact(&mut buf).unwrap();
    i16::from_le_bytes(buf)
}

pub fn read_le_u32_array<R: BufRead, const N: usize>(reader: &mut R) -> [u32; N] {
    let mut resp = [0u32; N];
    let mut buf = [0u8; 4];
    for i in 0..N {
        reader.read_exact(&mut buf).expect("Failed to read");
        resp[i] = u32::from_le_bytes(buf);
    }
    resp
}

pub fn read_fill_le_f32_array<R: BufRead>(reader: &mut R, array: &mut [f32]) {
    let mut buf = [0u8; 4];

    for i in 0..array.len() {
        reader.read_exact(&mut buf).unwrap();
        array[i] = f32::from_le_bytes(buf);
    }
}

pub fn read_fill_le_u16_array<R: BufRead>(reader: &mut R, array: &mut [u16]) {
    let mut buf = [0u8; 2];

    for i in 0..array.len() {
        reader.read_exact(&mut buf).unwrap();
        array[i] = u16::from_le_bytes(buf);
    }
}

// Helper function to write 64-bit value as two 32-bit integers
pub fn write_u64_as_i32s(header: &mut [i32], offset: usize, value: u64) {
    let bytes: [i32; 2] = bytemuck::cast(value);
    header[offset] = bytes[0];
    header[offset + 1] = bytes[1];
}

/// Find the maximum step number from DONE files in the output log directory.
/// 
/// This function searches for files named "DONE_<step>" in the given directory
/// and returns the highest step number found. Returns -1 if no DONE files are found
/// or if the directory cannot be read.
pub fn find_max_step(output_log_dir: Option<&str>) -> i32 {
    if output_log_dir.is_none() {
        return -1;
    }
    let output_log_dir = output_log_dir.unwrap();
    let path = Path::new(output_log_dir);
    
    // Check if the directory exists and is readable
    if !path.exists() || !path.is_dir() {
        return -1;
    }
    
    let mut max_step = -1;

    // Read directory entries
    let entries = match fs::read_dir(path) {
        Ok(entries) => entries,
        Err(_) => return -1,
    };
    
    for entry in entries {
        let entry = entry.unwrap();
        let file_name = entry.file_name();
        let file_name_str = file_name.to_string_lossy();
        
        // Extract the step number from files starting with "DONE_"
        if let Some(step_str) = file_name_str.strip_prefix("DONE_") {
            let step = step_str.parse::<i32>().unwrap();
            if step > max_step {
                max_step = step;
            }
        }
    }
    
    max_step
}
