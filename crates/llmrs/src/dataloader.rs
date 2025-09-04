#![allow(clippy::needless_range_loop)]

use std::fs::File;
use glob::glob;
use std::path::PathBuf;
use std::io::{BufReader, Seek, SeekFrom};
use core::mem::size_of;

use crate::random::{init_identity_permutation, Mt19937, random_permutation};
use crate::utils::{read_fill_le_u16_array, read_le_u32_array, read_le_u16};


const HEADER_SIZE: usize = 256;

/// Struct to hold all fields required to save/load the shuffling state of the dataloader
#[derive(Clone)]
pub struct ShufflingState {
    pub total_files: usize,
    pub shard_indices: Vec<i32>,
    pub shard_num_samples: usize,
    pub intra_shard_indices: Vec<i32>,
    pub restored_rng: Mt19937,
}

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
    buffer: Vec<i32>, // we fread data from file into this buffer
    // random shuffle related variables
    shuffle_rng: Option<Mt19937>,
    pub should_shuffle: bool,
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
            let shuffle_rng = Mt19937::new(42 + process_rank as u32);
            let mut shard_indices = vec![0usize;files.len()];
            init_identity_permutation(&mut shard_indices);
            (Some(shuffle_rng), Some(shard_indices))
        } else {
            (None, None)
        };

        // allocate all the space we'll need
        let buffer = vec![0i32; B * T +1];

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
            assert!(shard_ntok > num_processes * B * T);
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
    pub fn inputs(&self) -> &[i32] {
        &self.buffer[0..self.buffer.len() - 1]
    }

    /// expose targets as a non mutable slice of buffer
    pub fn targets(&self) -> &[i32] {
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
            self.buffer[i] = val as i32;
        }
        

    }

    pub fn resume_shuffling(&mut self, shuffling_state: ShufflingState) {
        self.should_shuffle = true;
        assert_eq!(shuffling_state.total_files, self.files.len());
        self.shard_indices = Some(shuffling_state.shard_indices.into_iter().map(|x| x as usize).collect());
        // FIXME from original code, but we are not guaranteed to have loaded the same shard.
        //assert_eq!(shuffling_state.shard_num_samples, self.shard_num_samples);
        self.intra_shard_indices = Some(shuffling_state.intra_shard_indices.into_iter().map(|x| x as usize).collect());
        self.shuffle_rng = Some(shuffling_state.restored_rng)
    }

    /// Save the current shuffling state for later restoration
    pub fn save_shuffling_state(&self) -> Option<ShufflingState> {
        if !self.should_shuffle {
            return None;
        }
        
        Some(ShufflingState {
            total_files: self.files.len(),
            shard_indices: self.shard_indices.as_ref()?.iter().map(|&x| x as i32).collect(),
            shard_num_samples: self.shard_num_samples,
            intra_shard_indices: self.intra_shard_indices.as_ref()?.iter().map(|&x| x as i32).collect(),
            restored_rng: self.shuffle_rng.as_ref()?.clone(),
        })
    }

    /// Resume the dataloader to a specific position
    pub fn resume(&mut self, current_shard_idx: usize, current_sample_idx: usize) {
        self.current_shard_idx = current_shard_idx;
        self.current_sample_idx = current_sample_idx;
        self.load_shard(current_shard_idx);
    }

} 


// ----------------------------------------------------------------------------
// Distributed Eval Loader
// Many evals (like) HellaSwag and MMLU are multiple-choice
// where there are 4 possible continuations and a label for the correct one
// We want to load and serve these style of evals
/*
Copy pasting the section on the eval datafile format, from data_common.py:
- First comes a header with 256 int32s
- The examples follow, each example is a stream of uint16_t:
    - <START_EXAMPLE> delimiter of 2**16-1, i.e. 65,535
    - <EXAMPLE_BYTES>, bytes encoding this example, allowing efficient skip to next
    - <EXAMPLE_INDEX>, the index of the example in the dataset
    - <LABEL>, the index of the correct completion
    - <NUM_COMPLETIONS>, indicating the number of completions (usually 4)
    - <NUM><CONTEXT_TOKENS>, where <NUM> is the number of tokens in the context
    - <NUM><COMPLETION_TOKENS>, repeated NUM_COMPLETIONS times
*/

// for now, could relax later
const ASSUMED_NUM_COMPLETIONS: usize = 4;

#[allow(non_snake_case)]
pub struct EvalLoader {
    // Configuration
    pub process_rank: usize,
    pub num_processes: usize,
    pub B: usize,
    pub T: usize,
    
    // File handling
    pub eval_file: Option<BufReader<File>>,
    pub num_examples: usize,
    
    // Batch management
    pub num_batches: usize,
    pub start_example_index: usize,
    pub end_example_index: usize,
    pub current_example_index: usize,
    
    // Data buffers
    pub buffer: Vec<u16>,
    pub inputs: Vec<i32>,
    pub targets: Vec<i32>,
    pub mask: Vec<u8>,
    pub label: Vec<i32>,
    
    // Constants
    pub num_completions: usize,
}


impl EvalLoader {
    /// Initialize the EvalLoader with file and configuration
    #[allow(non_snake_case)]
    pub fn init(filename: &str, B: usize, T: usize, process_rank: usize, num_processes: usize) -> Self {
        // open the file and validate the header
        let eval_file = File::open(filename)
            .unwrap_or_else(|_| panic!("Error: cannot open eval file {:?}", filename));
        let mut eval_file = BufReader::new(eval_file);
        
        // validate the header
        let header: [u32; HEADER_SIZE] = read_le_u32_array::<_, HEADER_SIZE>(&mut eval_file);
        
        if header[0] != 20240522 { panic!("Bad magic in eval file"); }
        if header[1] != 1 { panic!("Bad version in data file"); }
        let num_examples = header[2] as usize; // number of examples in the file
        assert!(num_examples >= num_processes); // avoid headaches for now
        let longest_example_bytes = header[3] as usize; // longest example in the file
        // basic sensibility check we could relax later. but roughly each example
        // contains the prompt (or "context") and 4 completions, all of these have to be
        // up to T tokens, and their tokens are uint16_t (so 2 bytes/token).
        // There's a few more things in each example but they are minor.
        // So longest example should be roughly this. Just trying to make sure it's sensible.
        assert!(longest_example_bytes > 0 && longest_example_bytes < (1 + ASSUMED_NUM_COMPLETIONS) * T * 2);

        // allocate all the space we'll need
        let can_fit_examples = B / ASSUMED_NUM_COMPLETIONS;
        
        let mut eval_loader = Self {
            process_rank,
            num_processes,
            B,
            T,
            eval_file: Some(eval_file),
            num_examples,
            num_batches: 0,  // Will be set by reset()
            start_example_index: 0, // Will be set by reset()
            end_example_index: 0, // Will be set by reset()
            current_example_index: 0, // Will be set by reset()
            buffer: vec![0u16; longest_example_bytes / 2], // Convert bytes to u16 count
            inputs: vec![0i32; B * T], // ???
            targets: vec![0i32; B * T], // ???
            mask: vec![0u8; B * T], // ???
            label: vec![0i32; can_fit_examples],// ???
            num_completions: 0, // Will be set by next_example()
        };
        
        // reset the loader, to initialize it
        eval_loader.reset();

        eval_loader
    }
    
    /// Reset the EvalLoader to start from the beginning
    pub fn reset(&mut self) {
        // we have to be careful that each process starts at the correct offset.
        // For example if there are N examples in the file and 4 processes,
        // then process 0 should start at 0, process 1 at N/4, process 2 at N/2, etc.
        // determine how much work there is for all processes
        let examples_per_process = self.num_examples.div_ceil(self.num_processes);
        let can_fit_examples = self.B / ASSUMED_NUM_COMPLETIONS;
        if can_fit_examples == 0 {
            // this could be fixed in the future, but for now keeping it simple and throw error when B too low
            println!("HellaSwag EvalLoader: batch size {} is < {}", self.B, ASSUMED_NUM_COMPLETIONS);
            println!("---> HINT: Disable HellaSwag eval with -h 0, or increase batch size with -b");
            panic!();
        }
        self.num_batches = examples_per_process.div_ceil(can_fit_examples);
        // determine the start and end example indices for this process
        self.start_example_index = examples_per_process * self.process_rank;
        self.end_example_index = examples_per_process * (self.process_rank + 1);
        // crop the end example index to the total number of examples
        if self.end_example_index > self.num_examples {
            self.end_example_index = self.num_examples;
        }

        // now seek through the file to the start of that example
        // utilize <EXAMPLE_BYTES> for efficiency
        self.eval_file.as_mut().unwrap().seek(
            SeekFrom::Start((HEADER_SIZE * size_of::<u32>()) as u64)
        ).unwrap();

        for i in 0..self.start_example_index {
            let mut example_header = vec![0u16; 3];
            // read 3 uint16_t values: <START_EXAMPLE>, <EXAMPLE_BYTES>, <EXAMPLE_INDEX>
            read_fill_le_u16_array(self.eval_file.as_mut().unwrap(), &mut example_header);
            // validate the <START_EXAMPLE> delimiter
            assert!(example_header[0] == 65535); // <START_EXAMPLE> delimiter
            // validate the <EXAMPLE_INDEX>
            assert!(example_header[2] as usize == i); // <EXAMPLE_INDEX> should match the loop index
            // skip to the next example, keeping in mind that we already read the header
            let remaining_bytes = example_header[1] as usize - 3 * size_of::<u16>();
            assert!(remaining_bytes > 0); // we expect some bytes in the example
            self.eval_file.as_mut().unwrap().seek(
                SeekFrom::Current(remaining_bytes as i64)
            ).unwrap();
        }
        
        // now we are at the start of the example we want to start at, pointing at <START_EXAMPLE>
        self.current_example_index = self.start_example_index;
    }

    #[allow(non_snake_case)]
    fn next_example(&mut self, example_batch_index: usize) {
        // this function populates the inputs, targets, mask, and label fields for one example
        // because every (B,T) tensor can fit multiple examples and we want to take advantage,
        // we also pass in the example_batch_index to indicate which example in the batch we are loading
        // and each example takes up ASSUMED_NUM_COMPLETIONS rows in the batch
        let B = self.B;
        let T = self.T;
        let eval_file = self.eval_file.as_mut().unwrap();
        let batch_dim_offset = example_batch_index * ASSUMED_NUM_COMPLETIONS;
        // read the current example header
        let mut example_header = vec![0u16; 3];
        read_fill_le_u16_array(eval_file, &mut example_header);
        // validate the <START_EXAMPLE> delimiter
        assert!(example_header[0] == 65535); // <START_EXAMPLE> delimiter
        // validate the <EXAMPLE_INDEX>
        let example_header_2_usize = example_header[2] as usize;
        assert!(example_header_2_usize == self.current_example_index); // <EXAMPLE_INDEX> should match the loop index
        assert!(example_header_2_usize >= self.start_example_index && example_header_2_usize < self.end_example_index);
        
        // process the example label
        let label  = read_le_u16(eval_file) as i32;
        let can_fit_examples = B / ASSUMED_NUM_COMPLETIONS;
        assert!(label > 0 && (label as usize) < ASSUMED_NUM_COMPLETIONS);
        assert!(example_batch_index < can_fit_examples);
        self.label[example_batch_index] = label; // store for output
        
        // process the number of completions
        let num_completions = read_le_u16(eval_file) as usize;
        assert!(num_completions == ASSUMED_NUM_COMPLETIONS); // we expect 4 completions for now
        assert!(batch_dim_offset + num_completions <= B); // we expect to fit in the batch
        self.num_completions = num_completions; // store for output

        // process the context
        // the context is shared for all completions, so we insert it into all data rows equally
        let context_length = read_le_u16(eval_file) as usize;
        assert!(context_length > 0 && context_length < T); // context is non-empty and up to T
        let mut context = vec![0u16; context_length];
        read_fill_le_u16_array(eval_file, &mut context);
        for b in 0..num_completions {
            let boff = batch_dim_offset + b;
            for i in 0..context_length {
                self.inputs[boff * T + i] = context[i] as i32;
            }
        }

        // process the completions, insert them in their row, right after the (shared) context
        for c in 0..num_completions {
            let coff = batch_dim_offset + c;
            let completion_length = read_le_u16(eval_file) as usize;
            assert!(completion_length > 0 && context_length + completion_length < T); // things fit?
            
            let mut completion = vec![0u16; completion_length];
            read_fill_le_u16_array(eval_file, &mut completion);

            for (i, tok_cur) in completion.iter().enumerate() {
                let tok_cur = *tok_cur as i32;
                // at inputs, the completions simply follow the context
                self.inputs[coff * T + context_length + i] = tok_cur;
                // at targets things start to get tricky
                // we expect the last context token to predict the first completion token
                // and then onwards from there.
                self.targets[coff * T + context_length + i -1] = tok_cur;
                // and at these positions, we want to set mask=1, because these are the
                // positions where we want to average the loss, in each row, to determine
                // its overall probability of following the context.
                self.mask[coff * T + context_length + i - 1] = 1;
            }
        }

        // advance the current example to point to the next one we'd load
        self.current_example_index += 1;

    }

    #[allow(non_snake_case)]
    pub fn next_batch(&mut self) {
        let B = self.B;
        
        // init mask to zeros, no need to do it for inputs & targets, the values where the mask
        // is set will be correctly overwritten every time.
        self.mask.fill(0);
        // ok here is the problem we are solving
        // we have a batch dimension of B, which we want to take full advantage of
        // each example has some number of completions (usually 4)
        // so we want to pack as many examples into rows of B as we can fit
        let can_fit_examples = B / ASSUMED_NUM_COMPLETIONS; // how many examples can we fit in the batch?

        for i in 0..can_fit_examples {
            if self.current_example_index >= self.end_example_index {
                break; // this process has exhausted its work, noop from here on
            }
            self.next_example(i);
        }

    }

    #[allow(non_snake_case)]
    pub fn stat_losses(&self, losses: &[f32]) -> i32 {
        // compute statistics of losses (B*T) resulting from a forward pass
        // on a batch that was constructed from EvalLoader
        // putting this functionality here because it is tightly coupled
        // with how we construct and represent the data batches.
        // returns the number of correct examples in this batch.
        
        let mut correct = 0;
        let B = self.B;
        let T = self.T;
        // iterate the examples in this batch
        let can_fit_examples = B / ASSUMED_NUM_COMPLETIONS;
        for i in 0..can_fit_examples {
            let mut min_loss = 0.0f32;
            let mut min_loss_index = -1;
            let mut active = false; // is this example active or fully empty?
            // iterate the completions in this example
            for b in 0..ASSUMED_NUM_COMPLETIONS {
                let boff = i * ASSUMED_NUM_COMPLETIONS + b;
                // evaluate the quality of this completion
                // its quality is simply the average loss over the tokens
                let mut average_loss = 0.0f32;
                let mut count = 0;
                for t in 0..T {
                    let mask = self.mask[boff * T + t];
                    if mask == 1 {
                        active = true;
                        average_loss += losses[boff * T + t];
                        count += 1;
                    }
                }
                if count > 0 { 
                    average_loss /= count as f32; 
                }
                if b == 0 || average_loss < min_loss {
                    min_loss = average_loss;
                    min_loss_index = b as i32;
                }
            }
            if active && (min_loss_index == self.label[i]) {
                correct += 1;
            }
        }
        correct
    }

} 

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;


    #[test]
    fn test_shuffling_state_save_and_restore() {
        // Print current directory for debugging
        println!("Current directory: {}", env::current_dir().unwrap().display());

        // Create a simple dataloader with shuffling enabled
        let dataloader = Dataloader::new("../../dev/data/tinyshakespeare/*.bin", 4, 8, true);
        
        // Save the shuffling state
        let saved_state = dataloader.save_shuffling_state();
        assert!(saved_state.is_some());
        
        let shuffling_state = saved_state.unwrap();
        
        // Verify the saved state has the correct values
        assert_eq!(shuffling_state.total_files, dataloader.files.len());
        assert_eq!(shuffling_state.shard_num_samples, dataloader.shard_num_samples);
        assert!(!shuffling_state.shard_indices.is_empty());
        assert!(!shuffling_state.intra_shard_indices.is_empty());
        
        // Create a new dataloader and restore the state
        let mut new_dataloader = Dataloader::new("../../dev/data/tinyshakespeare/*.bin", 4, 8, false);
        new_dataloader.resume_shuffling(shuffling_state);
        
        // Verify the state was restored correctly
        assert!(new_dataloader.should_shuffle);
        assert!(new_dataloader.shuffle_rng.is_some());
        assert!(new_dataloader.shard_indices.is_some());
        assert!(new_dataloader.intra_shard_indices.is_some());
    }

    #[test]
    fn test_shuffling_state_no_shuffle() {
        // Create a dataloader without shuffling
        let dataloader = Dataloader::new("../../dev/data/tinyshakespeare/*.bin", 4, 8, false);
        
        // Try to save shuffling state - should return None
        let saved_state = dataloader.save_shuffling_state();
        assert!(saved_state.is_none());
    }
} 