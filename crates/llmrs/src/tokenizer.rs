// Tokenizer module skeleton

use std::fs::File;
use std::io::Read;

pub struct Tokenizer {
    _vocab_size: u32,
    token_table: Vec<String>,
    pub eot_token: i32
}

impl Tokenizer {
    pub fn init(filename: &str) -> Self {
        let mut file = File::open(filename).expect("Failed to open tokenizer model file");

        let mut header: [u32; 256] = [0; 256];

        // Explicitly read Little Endian
        let mut buf = [0u8; 4];
        for i in 0..header.len() {
            file.read_exact(&mut buf).expect("Failed to read header");
            header[i] = u32::from_le_bytes(buf);
        }
        
        assert_eq!(header[0], 20240328);
        let version: u32 = header[1];
        let vocab_size: u32 = header[2];
        let eot_token: i32;
        match version {
            1 => {
                // version 1 didn't include the EOT token id
                // so we assume it is 50256, the EOT in GPT-2
                assert_eq!(vocab_size, 50257); // let's be defensive here
                eot_token = 50256;
            }
            2 => {
                eot_token = header[3] as i32;
            }
            _ => {
                panic!("Tokenizer model file {} has bad version: {}", filename, version);
            }
        }

        // read in all the tokens
        let mut token_table: Vec<String> = Vec::new();
        for _ in 0..vocab_size {
            let mut len = [0u8];
            file.read_exact(&mut len).expect("Failed to read token length");
            let len = len[0];
            assert!(len > 0); // every token should be at least one character

            let mut token_bytes = vec![0u8; len as usize];
            file.read_exact(&mut token_bytes).expect("Failed to read token");

            let token = match String::from_utf8(token_bytes) {
                Ok(s) => s,
                Err(_) => "�".to_string()
            };
            token_table.push(token);
        }
        
        Self {
            _vocab_size: vocab_size,
            token_table,
            eot_token,
        }
    }

    pub fn decode(&self, token_id: i32) -> String {
        self.token_table[token_id as usize].clone()
        
    }
} 