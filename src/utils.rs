use std::io::BufRead;


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