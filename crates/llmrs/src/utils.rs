use std::io::{BufRead, Read, Write};
use std::fs;
use std::path::Path;
use cust::memory::DeviceCopy;
use cust::{
    memory::{DeviceSlice, LockedBuffer, AsyncCopyDestination},
    stream::{Stream},
};
use std::cmp::min;
use std::mem;
use bytemuck::{NoUninit, AnyBitPattern};

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

pub fn read_fill_le_f32_array<R: BufRead>(reader: &mut R, array: &mut Vec<f32>) {
    let mut buf = [0u8; 4];

    for i in 0..array.len() {
        reader.read_exact(&mut buf).unwrap();
        array[i] = f32::from_le_bytes(buf);
    }
}

pub fn read_fill_le_u16_array<R: BufRead>(reader: &mut R, array: &mut Vec<u16>) {
    let mut buf = [0u8; 2];

    for i in 0..array.len() {
        reader.read_exact(&mut buf).unwrap();
        array[i] = u16::from_le_bytes(buf);
    }
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

// ----------------------------------------------------------------------------
// Utilities to Read & Write between CUDA memory <-> files

pub fn file_to_device<T: DeviceCopy + NoUninit + AnyBitPattern, R: Read>(dst: &mut DeviceSlice<T>, 
                      file_reader: &mut R, 
                      buffer_size: usize, 
                      stream: &Stream) {

    // 1) Allocate pinned host memory (page-locked). Using `uninitialized` so we can fill by reading.
    let mut buffer_space = unsafe { LockedBuffer::<T>::uninitialized(2 * buffer_size).unwrap() };
    let (mut read_buffer, mut write_buffer) = buffer_space.as_mut_slice().split_at_mut(buffer_size);

    // 2) Prime pipeline: read first chunk
    let mut copy_amount = min(buffer_size, dst.len());
    let bytes = bytemuck::cast_slice_mut(&mut read_buffer[..copy_amount]);
    file_reader.read_exact(bytes).unwrap();

    let mut rest_bytes = dst.len() - copy_amount;
    let mut write_len = copy_amount;
    // Swap roles (read_buf will be filled while write_buf is copied)
    mem::swap(&mut read_buffer, &mut write_buffer);

    let mut dst_offset = 0isize;

    // 3) now the main loop; as long as there are bytes left
    while rest_bytes > 0 {
        unsafe {
            let write_ptr = dst.as_device_ptr().offset(dst_offset);
            let mut write_slice = DeviceSlice::from_raw_parts_mut(write_ptr, write_len);
            write_slice.async_copy_from(&write_buffer[..write_len], stream).unwrap();
        }
        
        // while this is going on, read from disk
        copy_amount = min(buffer_size, rest_bytes);
        let bytes = bytemuck::cast_slice_mut(&mut read_buffer[..copy_amount]);
        file_reader.read_exact(bytes).unwrap();
        stream.synchronize().unwrap();

        dst_offset += write_len as isize;
        mem::swap(&mut read_buffer, &mut write_buffer);
        rest_bytes -= copy_amount;
        write_len = copy_amount;
    }

    // 4) copy the last remaining write buffer to gpu
    unsafe {
        let write_ptr = dst.as_device_ptr().offset(dst_offset);
        let mut write_slice = DeviceSlice::from_raw_parts_mut(write_ptr, write_len);
        write_slice.async_copy_from(&write_buffer[..write_len], stream).unwrap();
    }
    stream.synchronize().unwrap();

}

pub fn device_to_file(file: &mut std::fs::File, src: &DeviceSlice<f32>, buffer_size: usize, stream: &Stream) {

    // 1) Allocate pinned host memory (page-locked). Using `uninitialized` so we can fill by reading.
    let mut buffer_space = unsafe { LockedBuffer::<f32>::uninitialized(2 * buffer_size).unwrap() };
    let (mut read_buffer, mut write_buffer) = buffer_space.as_mut_slice().split_at_mut(buffer_size);

    // 2) Prime pipeline: copy first chunk from device
    let mut copy_amount = min(buffer_size, src.len());
    unsafe {
        let read_ptr = src.as_device_ptr();
        let read_slice = DeviceSlice::from_raw_parts(read_ptr, copy_amount);
        // async_copy_to expects both buffers to be of the same size, force read_buffer to be copy_amount long

        read_slice.async_copy_to(&mut read_buffer[..copy_amount], stream)
            .map_err(|e| {
                eprintln!("CUDA async_copy_to failed with error: {:?}", e);
                eprintln!("copy_amount: {}, buffer_size: {}, src_len: {}", copy_amount, buffer_size, src.len());
                eprintln!("read_buffer len: {}, read_slice len: {}", read_buffer.len(), read_slice.len());
                e
            }).unwrap();
    }
    stream.synchronize().unwrap();

    let mut rest_bytes = src.len() - copy_amount;
    let mut write_len = copy_amount;
    // Swap roles (read_buf will be filled while write_buf is written to file)
    mem::swap(&mut read_buffer, &mut write_buffer);

    let mut src_offset = copy_amount as isize;

    // 3) now the main loop; as long as there are bytes left
    while rest_bytes > 0 {
        // Write the current buffer to file
        let bytes = bytemuck::cast_slice(&write_buffer[..write_len]);
        file.write_all(bytes).unwrap();
        
        // while this is going on, copy from device
        copy_amount = min(buffer_size, rest_bytes);
        unsafe {
            let read_ptr = src.as_device_ptr().offset(src_offset);
            let read_slice = DeviceSlice::from_raw_parts(read_ptr, copy_amount);
            read_slice.async_copy_to(&mut read_buffer[..copy_amount], stream).unwrap();
        }
        stream.synchronize().unwrap();

        src_offset += copy_amount as isize;
        mem::swap(&mut read_buffer, &mut write_buffer);
        rest_bytes -= copy_amount;
        write_len = copy_amount;
    }

    // 4) write the last remaining write buffer to file
    let bytes = bytemuck::cast_slice(&write_buffer[..write_len]);
    file.write_all(bytes).unwrap();
}


#[cfg(test)]
mod tests {
    use crate::utils::{device_to_file, file_to_device};
    use std::fs;
    use std::io::Seek;
    use cust::{
        memory::DeviceBuffer,
        stream::Stream,
        prelude::*,
    };
    use rand::{Rng, SeedableRng};
    

    fn test_device_file_io(nelem: usize, wt_buf_size: usize, rd_buf_size: usize) {
        // Initialize CUDA context
        let _ctx = cust::quick_init().unwrap();
        
        let mut data = DeviceBuffer::<f32>::zeroed(nelem).unwrap();

        // generate random array
        let mut random_data = vec![0.0f32; nelem];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let dist = rand::distributions::Uniform::new(-100.0f32, 100.0f32);
        for i in 0..nelem {
            random_data[i] = rng.sample(dist);
        }

        data.copy_from(&random_data).unwrap();

        let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();

        // open R/W and truncate/create
        let mut tmp = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open("tmp.bin")
            .unwrap();
        
        device_to_file(&mut tmp, &data, wt_buf_size, &stream);
        
        // make sure data + metadata are flushed
        tmp.sync_all().unwrap();
        // rewind and read file -> device. No close/reopen because for some reason the file will not appear on the file system immediately
        tmp.rewind().unwrap();

        let mut reload = DeviceBuffer::<f32>::zeroed(nelem).unwrap();
        
        file_to_device(&mut reload, &mut tmp, rd_buf_size, &stream);
        drop(tmp);

        let mut cmp = vec![0.0f32; nelem];
        reload.copy_to(&mut cmp).unwrap();
        for i in 0..nelem {
            if random_data[i] != cmp[i] {
                let _ = fs::remove_file("tmp.bin");
                panic!("FAIL: Mismatch at position {}: {} vs {}", i, random_data[i], cmp[i]);
            }
        }

        let _ = fs::remove_file("tmp.bin");
    }

    #[test]
    fn test_device_file_io_buffers_larger_than_data() {
        test_device_file_io(1025, 10000, 10000);
    }

    #[test]
    fn test_device_file_io_different_and_smaller_buffers() {
        test_device_file_io(1025, 1024, 513);
    }

    #[test]
    fn test_device_file_io_exact_match() {
        test_device_file_io(500, 500 * std::mem::size_of::<f32>(), 500 * std::mem::size_of::<f32>());
    }

    #[test]
    fn test_device_file_io_large_array() {
        test_device_file_io(125000, 10000, 10000);
    }

}

