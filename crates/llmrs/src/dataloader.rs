use std::fs::File;
use glob::glob;
use std::path::PathBuf;
use std::io::{BufReader, Seek, SeekFrom};
use core::mem::size_of;
use rand_mt::Mt;

use crate::random::{init_identity_permutation, random_permutation};
use crate::utils::{read_fill_le_u16_array, read_le_u32_array};


const HEADER_SIZE: usize = 256;

pub struct Dataloader {
    files: Vec<PathBuf>,
    // batch and token information
    //B: usize,
    //T: usize,
    pub num_tokens: usize,   // total number of tokens
    shard_num_samples: usize, // total number of samples in the current shard per process
    // shards and current position
    current_shard_idx: usize,
    current_sample_idx: usize,
    // file handle
    tokens_file: Option<BufReader<File>>,
    // data buffers
    buffer: Vec<u32>, // we fread data from file into this buffer
    // random shuffle related variables
    shuffle_rng: Option<Mt>,
    should_shuffle: bool,
    shard_indices: Option<Vec<usize>>,
    intra_shard_indices: Option<Vec<usize>>,
    // sizes in bytes
    total_batch_size_bytes: usize,  // total across all processes
    local_batch_offset_bytes: usize,  // inner-sample offset for this process
    header_bytes: usize  // header size in bytes

}

impl Dataloader {

    #[allow(non_snake_case)]
    pub fn new(file_pattern: &str, B: usize, T: usize, should_shuffle: bool) -> Self {

        let num_processes: usize = 1;
        let process_rank = 0;

        let files: Vec<PathBuf> = glob(file_pattern)
            .expect("Error: failed to glob pattern")
            .filter_map(Result::ok)
            .collect();

        if files.is_empty() {
            panic!("Error: no files found matching the pattern: {}", file_pattern);
        }

        let (shuffle_rng, shard_indices) =  if should_shuffle {
            let shuffle_rng = Mt::new(42 + process_rank as u32);
            let mut shard_indices = vec![0usize;files.len()];
            init_identity_permutation(&mut shard_indices);
            (Some(shuffle_rng), Some(shard_indices))
        } else {
            (None, None)
        };

        // allocate all the space we'll need
        let buffer = vec![0u32; B * T +1];

        let mut dataloader = Self {
            files,
            //B,
            //T,
            num_tokens: 0,
            shard_num_samples: 0,
            current_shard_idx: 0,
            current_sample_idx: 0,
            tokens_file: None,
            buffer,
            shuffle_rng,
            should_shuffle,
            shard_indices,
            intra_shard_indices: None,
            header_bytes: HEADER_SIZE * std::mem::size_of::<u32>(),
            total_batch_size_bytes: num_processes * B * T * std::mem::size_of::<u16>(),
            local_batch_offset_bytes: process_rank * B * T * std::mem::size_of::<u16>(),
        };

        // inspect and validate all shards so we don't get any runtime errors later
        // if too slow / too many shards, may wish to revisit later
        let mut ntok_total: usize = 0;
        for shard_index in 0..dataloader.files.len() {
            let shard_ntok = dataloader.load_shard(shard_index);
            // we need at least one batch/shard, the way things are written right now.
            // can be relaxed a lot later.
            assert!(shard_ntok >= num_processes * B * T + 1);
            ntok_total += shard_ntok;
        }
        println!("DataLoader: filename_pattern: {}", file_pattern);
        println!("DataLoader: Found {} tokens across {} shards\n", ntok_total, dataloader.files.len());
        dataloader.num_tokens = ntok_total;

        dataloader.reset();

        dataloader        
    }

    pub fn next_batch(&mut self) {
        // if the next batch would go past the end of the file, advance the loader
        if self.current_sample_idx >= self.shard_num_samples {
            self.advance();
        }
        self.load_batch();
        self.current_sample_idx += 1;
    }

    fn prepare_intra_shard_indices(&mut self){
        // shuffle the examples inside the shards
        let mut intra_shard_indices = vec![0usize; self.shard_num_samples];
        let shuffle_rng = self.shuffle_rng.as_mut().unwrap();
        init_identity_permutation(&mut intra_shard_indices);
        random_permutation(&mut intra_shard_indices, self.shard_num_samples, shuffle_rng);
        self.intra_shard_indices = Some(intra_shard_indices);
    }

    pub fn reset(&mut self) {
        self.current_shard_idx = 0;
        self.current_sample_idx = 0;

        if self.should_shuffle {
            let shuffle_rng = self.shuffle_rng.as_mut().unwrap();
            let shard_indices = self.shard_indices.as_mut().unwrap();
            random_permutation(shard_indices, self.files.len(), shuffle_rng);
        }
        self.load_shard(self.current_shard_idx);

        if self.should_shuffle {
            self.prepare_intra_shard_indices();
        }
    }

    /// expose inputs as a non mutable slice of buffer
    pub fn inputs(&self) -> &[u32] {
        &self.buffer[0..self.buffer.len() - 1]
    }

    /// expose targets as a non mutable slice of buffer
    pub fn targets(&self) -> &[u32] {
        &self.buffer[1..self.buffer.len()]
    }

    fn load_shard(&mut self, shard_index: usize) -> usize {
        let mut shard_index = shard_index;
        if self.should_shuffle {
            let shard_indices = self.shard_indices.as_ref().unwrap();
            shard_index = shard_indices[shard_index];
        }

        let filename = &self.files[shard_index];

        let file = File::open(filename)
            .unwrap_or_else(|_| panic!("Error: cannot open file {:?}", filename));
        let mut tokens_file_reader = BufReader::new(file);

        // validate the header
        let header: [u32; HEADER_SIZE] = read_le_u32_array::<_, HEADER_SIZE>(&mut tokens_file_reader);

        if header[0] != 20240520 {
            println!("Bad magic in the data file");
            println!("---> HINT: Are you passing in a correct file?");
            println!("---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.");
            panic!()
        }
        if header[1] != 1 { panic!("Bad version in data file");  }
        let ntok = header[2] as usize; // number of tokens in the file
        assert!(ntok > 0); // we expect some tokens in the file. this should never trip, right?
        // determine the file size and make sure it is consistent with the number of tokens
        let file_size = std::fs::metadata(filename).expect("Unable to get file metadata").len();
        let exptected_file_size = header.len() * size_of::<u32>() + ntok  * size_of::<u16>();
        // we expect ntok in the file to be consistent with filesize, assert that is the case
        if file_size != exptected_file_size as u64 {
            panic!("Error: file size is not as expected")
        }

        // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
        self.shard_num_samples = (ntok - 1) * std::mem::size_of::<u16>() / self.total_batch_size_bytes;

        self.tokens_file = Some(tokens_file_reader);

        ntok
    }

    fn advance(&mut self) {
        if self.current_shard_idx == self.files.len() - 1 {
            // if we are at the last shard, we reset the loader and start a new epoch
            self.reset();
            return;
        }

        // advance the loader by loading the next data shard and resetting the position
        self.current_shard_idx += 1;
        self.current_sample_idx = 0;
        self.load_shard(self.current_shard_idx);

        if self.should_shuffle {
            self.prepare_intra_shard_indices();
        }
    }

    fn load_batch(&mut self) {
        assert!(!self.should_shuffle || self.intra_shard_indices.is_some());
        assert!(self.current_sample_idx < self.shard_num_samples);

        let idx = if self.should_shuffle { 
            self.intra_shard_indices.as_ref().unwrap()[self.current_sample_idx]
        } else {
            self.current_sample_idx
        };
        let global_batch_offset_bytes = idx * self.total_batch_size_bytes;
        let current_offset = self.header_bytes + global_batch_offset_bytes + self.local_batch_offset_bytes;

        self.tokens_file.as_mut().unwrap().seek(
            SeekFrom::Start(current_offset as u64)
        ).unwrap();

        let mut u16_buffer = vec![0; self.buffer.len()];
        read_fill_le_u16_array(self.tokens_file.as_mut().unwrap(), &mut u16_buffer);
        for (i, &val) in u16_buffer.iter().enumerate() {
            self.buffer[i] = val as u32;
        }
        

    }


} 

/// Dummy EvalLoader for multiple-choice evaluation datasets (e.g. HellaSwag)
/// This is a placeholder implementation that matches the C interface
pub struct EvalLoader {
    // Configuration
    pub process_rank: i32,
    pub num_processes: i32,
    pub B: usize,
    pub T: usize,
    
    // File handling
    pub eval_file: Option<std::fs::File>,
    pub num_examples: i32,
    
    // Batch management
    pub num_batches: i32,
    pub start_example_index: i32,
    pub end_example_index: i32,
    pub current_example_index: i32,
    
    // Data buffers
    pub buffer: Vec<u16>,
    pub inputs: Vec<i32>,
    pub targets: Vec<i32>,
    pub mask: Vec<u8>,
    pub label: Vec<i32>,
    
    // Constants
    pub num_completions: i32,
}

impl Default for EvalLoader {
    fn default() -> Self {
        Self {
            process_rank: 0,
            num_processes: 1,
            B: 1,
            T: 1024,
            eval_file: None,
            num_examples: 0,
            num_batches: 0,
            start_example_index: 0,
            end_example_index: 0,
            current_example_index: 0,
            buffer: Vec::new(),
            inputs: Vec::new(),
            targets: Vec::new(),
            mask: Vec::new(),
            label: Vec::new(),
            num_completions: 4, // ASSUMED_NUM_COMPLETIONS from C code
        }
    }
}

impl EvalLoader {
    /// Initialize the EvalLoader with file and configuration
    pub fn init(&mut self, filename: &str, B: usize, T: usize, process_rank: i32, num_processes: i32) {
        self.process_rank = process_rank;
        self.num_processes = num_processes;
        self.B = B;
        self.T = T;
        
        // Dummy implementation - just set some placeholder values
        self.num_examples = 1000; // placeholder
        self.num_batches = 10; // placeholder
        
        // Allocate buffers
        let can_fit_examples = (B / self.num_completions as usize).max(1);
        self.buffer = vec![0u16; 1024]; // placeholder size
        self.inputs = vec![0i32; B * T];
        self.targets = vec![0i32; B * T];
        self.mask = vec![0u8; B * T];
        self.label = vec![0i32; can_fit_examples];
        
        self.reset();
    }
    
    /// Reset the EvalLoader to start from the beginning
    pub fn reset(&mut self) {
        // Dummy implementation - just reset indices
        self.current_example_index = self.start_example_index;
    }
    
    /// Load the next batch of examples
    pub fn next_batch(&mut self) {
        // Dummy implementation - just increment example index
        self.current_example_index += 1;
        if self.current_example_index >= self.end_example_index {
            self.current_example_index = self.start_example_index;
        }
    }
    
    /// Calculate statistics from losses and return number of correct predictions
    pub fn stat_losses(&self, losses: &[f32]) -> i32 {
        // Dummy implementation - return random correct count
        (self.num_examples / 4) as i32 // placeholder: 25% accuracy
    }
    
    /// Free resources (no-op in Rust due to RAII)
    pub fn free(&mut self) {
        // Rust handles cleanup automatically
    }
} 