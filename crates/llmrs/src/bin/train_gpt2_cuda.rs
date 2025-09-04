#[cfg(feature = "cuda")]
use llm_rs::common::{vec_f32_to_floatx, zero_floatx, FloatX, PrecisionMode, PRECISION_MODE, ToF32};
#[cfg(feature = "cuda")]
use clap::Parser;
#[cfg(feature = "cuda")]
use std::io::{Read, BufReader, Write};
#[cfg(feature = "cuda")]
use std::fs::File;
#[cfg(feature = "cuda")]
use std::cmp::{max, min};
#[cfg(feature = "cuda")]
use llm_rs::utils::{find_max_step, read_le_u32_array, write_u64_as_i32s};
#[cfg(feature = "cuda")]
use llm_rs::cuda_utils::file_to_device;
#[cfg(feature = "cuda")]
use llm_rs::{dataloader::Dataloader, dataloader::EvalLoader, tokenizer::Tokenizer, scheduler::LearningRateScheduler};
#[cfg(feature = "cuda")]
use llm_rs::logger::Logger;
#[cfg(feature = "cuda")]
use llm_rs::sampler;
#[cfg(feature = "cuda")]
use llm_rs::cuda_launchers::*;
#[cfg(feature = "cuda")]
use llm_rs::outlier_detector::OutlierDetector;
#[cfg(feature = "cuda")]
use cust::{prelude::*};
#[cfg(feature = "cuda")]
use cust::memory::{DeviceCopy, GpuBuffer, LockedBuffer};

#[cfg(feature = "cuda")]
mod train_gpt2_cuda {
use super::*;

// Simple MultiGpuConfig struct for checkpoint functionality
#[derive(Debug, Clone)]
struct MultiGpuConfig {
    process_rank: i32,
    num_processes: usize,
    zero_stage: i32,
}

impl MultiGpuConfig {
    fn new(process_rank: i32, num_processes: usize, zero_stage: i32) -> Self {
        Self {
            process_rank,
            num_processes,
            zero_stage,
        }
    }
}

#[derive(Debug)]
struct GPT2Config {
    max_seq_len: usize, // max sequence length, e.g. 1024
    vocab_size: usize, // vocab size, e.g. 50257
    padded_vocab_size: usize, // padded to e.g. %128==0, 50304
    num_layers: usize, // number of layers, e.g. 12
    num_heads: usize,  // number of heads in attention, e.g. 12
    channels: usize, // number of channels, e.g. 768
}

// both GPT-2 and GPT-3 use the same tokenizer with 50257 tokens
const VOCAB_SIZE: usize = 50257;
const PADDED_VOCAB_SIZE: usize = 50304; // padded to 128 for CUDA kernel efficiency

// the parameters of the model
pub const NUM_PARAMETER_TENSORS: usize = 16;

struct ParameterTensors {
    wte: DeviceSlice<FloatX>, // (Vp, C)
    wpe: DeviceSlice<FloatX>, // (maxT, C)
    ln1w: DeviceSlice<FloatX>, // (L, C)
    ln1b: DeviceSlice<FloatX>, // (L, C)
    qkvw: DeviceSlice<FloatX>, // (L, 3*C, C)
    qkvb: DeviceSlice<FloatX>, // (L, 3*C)
    attprojw: DeviceSlice<FloatX>, // (L, C, C)
    attprojb: DeviceSlice<FloatX>, // (L, C)
    ln2w: DeviceSlice<FloatX>, // (L, C)
    ln2b: DeviceSlice<FloatX>, // (L, C)
    fcw: DeviceSlice<FloatX>, // (L, 4*C, C)
    fcb: DeviceSlice<FloatX>, // (L, 4*C)
    fcprojw: DeviceSlice<FloatX>, // (L, C, 4*C)
    fcprojb: DeviceSlice<FloatX>, // (L, C)
    lnfw: DeviceSlice<FloatX>, // (C)
    lnfb: DeviceSlice<FloatX>, // (C)

    memory: DeviceBuffer<FloatX>,
    sizes: ParameterTensorsSizes,
    num_parameters: usize,
}

impl ParameterTensors {
    #[allow(non_snake_case)]
    fn allocate(config: &GPT2Config) -> Self {
        let C = config.channels;
        let L = config.num_layers;
        let Vp = config.padded_vocab_size;
        let seq_len = config.max_seq_len;
        let sizes = ParameterTensors::sizes(seq_len, C, L, Vp);

        let allocation = sizes.to_vec();
        assert_eq!(allocation.len(), NUM_PARAMETER_TENSORS);
        let total_size = allocation.iter().sum();
        let mut param_memory = DeviceBuffer::<FloatX>::zeroed(total_size).unwrap();
        let slices = buffer_to_slices(&allocation, &mut param_memory);

        Self {
            wte: slices[0],
            wpe: slices[1],
            ln1w: slices[2],
            ln1b: slices[3],
            qkvw: slices[4],
            qkvb: slices[5],
            attprojw: slices[6],
            attprojb: slices[7],
            ln2w: slices[8],
            ln2b: slices[9],
            fcw: slices[10],
            fcb: slices[11],
            fcprojw: slices[12],
            fcprojb: slices[13],
            lnfw: slices[14],
            lnfb: slices[15],
            memory: param_memory,
            sizes: sizes,
            num_parameters: total_size,
        }
    }

    #[allow(non_snake_case)]
    fn sizes(seq_len: usize, C: usize, L: usize, Vp: usize) -> ParameterTensorsSizes {
        ParameterTensorsSizes {
            wte: Vp * C,
            wpe: seq_len * C,
            ln1w: L * C,
            ln1b: L * C,
            qkvw: L * 3*C * C,
            qkvb: L * 3*C,
            attprojw: L * C * C,
            attprojb: L * C,
            ln2w: L * C,
            ln2b: L * C,
            fcw: L * 4*C * C,
            fcb: L * 4*C,
            fcprojw: L * C * 4*C,
            fcprojb: L * C,
            lnfw: C,
            lnfb: C,
        }
    }
}

struct ParameterTensorsSizes {
    wte: usize,
    wpe: usize,
    ln1w: usize,
    ln1b: usize,
    qkvw: usize,
    qkvb: usize,
    attprojw: usize,
    attprojb: usize,
    ln2w: usize,
    ln2b: usize,
    fcw: usize,
    fcb: usize,
    fcprojw: usize,
    fcprojb: usize,
    lnfw: usize,
    lnfb: usize,
}

impl ParameterTensorsSizes {
    fn to_vec(&self) -> Vec<usize> {
        vec![self.wte, self.wpe, self.ln1w, self.ln1b, self.qkvw, self.qkvb, self.attprojw, self.attprojb, self.ln2w, self.ln2b, self.fcw, self.fcb, self.fcprojw, self.fcprojb, self.lnfw, self.lnfb]
    }
}

// the parameters of the model
pub const NUM_ACTIVATION_TENSORS: usize = 21;
pub const NUM_ACTIVATION_TENSORS_FLOATX: usize = 14;
pub const NUM_ACTIVATION_TENSORS_FLOAT: usize = 7;

struct ActivationTensors {
    encoded: DeviceSlice<FloatX>, // (B, T, C)
    ln1: DeviceSlice<FloatX>,     // (L, B, T, C)
    ln1_mean: DeviceSlice<f32>,// (L, B, T)
    ln1_rstd: DeviceSlice<f32>,// (L, B, T)
    atty: DeviceSlice<FloatX>, // (L, B, T, C)
    att: DeviceSlice<FloatX>,  // (L, B, NH, T, T)
    residual2: DeviceSlice<FloatX>,  // (L, B, T, C)
    ln2: DeviceSlice<FloatX>,  // (L, B, T, C)
    ln2_mean: DeviceSlice<f32>, // (L, B, T)
    ln2_rstd: DeviceSlice<f32>, // (L, B, T)
    fch: DeviceSlice<FloatX>, // (L, B, T, 4*C)
    fch_gelu: DeviceSlice<FloatX>, // (L, B, T, 4*C)
    residual3: DeviceSlice<FloatX>,  // (L, B, T, C)
    lnf: DeviceSlice<FloatX>,  // (B, T, C);   if LN recomputation is enabled (-r 2 and above), will be used for _all_ layernorms
    lnf_mean: DeviceSlice<f32>,  // (B, T)
    lnf_rstd: DeviceSlice<f32>,  // (B, T)
    losses: DeviceSlice<f32>,  // (B, T), will be accumulated in micro-steps
    // adding these two compared to the CPU .c code, needed for attention kernel as buffers
    qkvr: DeviceSlice<FloatX>, // (L, B, T, 3*C)
    // in inference mode, this buffer will store the logits
    // in training mode, this buffer will contain the *gradients* of the logits.
    // during the processing of transformer blocks, we will also use this as a
    // general scratchpad buffer. Allocation is made large enough to hold (B, T, 3C),
    // (B, NH, T, T), and (B, T, V) shaped tensors.
    output: DeviceSlice<FloatX>,

    // some additional scratch buffers
    scratch_bt4c: DeviceSlice<FloatX>,   // (B, T, 4*C)
    scratch_btc: DeviceSlice<FloatX>,    // (B, T, C)

    memory_floatx: DeviceBuffer::<FloatX>,
    memory_float: DeviceBuffer::<f32>,
}

impl ActivationTensors {

    #[allow(non_snake_case)]
    fn allocate(config: &GPT2Config, B: usize, T: usize, recompute: i32) -> Self {
        let C = config.channels;
        let L = config.num_layers;
        let NH = config.num_heads;
        let Vp = config.padded_vocab_size;
        let sizes = ActivationTensors::sizes(B, T, C, L, NH, Vp, recompute);

        let floatx_allocation = [sizes.encoded, sizes.ln1, sizes.atty, sizes.att, sizes.residual2, sizes.ln2, sizes.fch, sizes.fch_gelu, sizes.residual3, sizes.lnf, sizes.qkvr, sizes.output, sizes.scratch_bt4c, sizes.scratch_btc];
        assert_eq!(floatx_allocation.len(), NUM_ACTIVATION_TENSORS_FLOATX);
        let floatx_total_size = floatx_allocation.iter().sum();
        let mut act_memory_floatx = DeviceBuffer::<FloatX>::zeroed(floatx_total_size).unwrap();
        let mut floatx_slices = buffer_to_slices(&floatx_allocation, &mut act_memory_floatx);
        assert_eq!(floatx_slices.len(), NUM_ACTIVATION_TENSORS_FLOATX);

        let f32_allocation = [sizes.ln1_mean, sizes.ln1_rstd, sizes.ln2_mean, sizes.ln2_rstd, sizes.lnf_mean, sizes.lnf_rstd, sizes.losses];
        assert_eq!(f32_allocation.len(), NUM_ACTIVATION_TENSORS_FLOAT);
        let f32_total_size = f32_allocation.iter().sum();
        let mut act_memory_f32 = DeviceBuffer::<f32>::zeroed(f32_total_size).unwrap();
        let mut f32_slices = buffer_to_slices(&f32_allocation, &mut act_memory_f32);
        assert_eq!(f32_slices.len(), NUM_ACTIVATION_TENSORS_FLOAT);
        
        Self {
            encoded: floatx_slices.remove(0),
            ln1: floatx_slices.remove(0),
            ln1_mean: f32_slices.remove(0),
            ln1_rstd: f32_slices.remove(0),
            atty: floatx_slices.remove(0),
            att: floatx_slices.remove(0),
            residual2: floatx_slices.remove(0),
            ln2: floatx_slices.remove(0),
            ln2_mean: f32_slices.remove(0),
            ln2_rstd: f32_slices.remove(0),
            fch: floatx_slices.remove(0),
            fch_gelu: floatx_slices.remove(0),
            residual3: floatx_slices.remove(0),
            lnf: floatx_slices.remove(0),
            lnf_mean: f32_slices.remove(0),
            lnf_rstd: f32_slices.remove(0),
            losses: f32_slices.remove(0),
            qkvr: floatx_slices.remove(0),
            output: floatx_slices.remove(0),
            scratch_bt4c: floatx_slices.remove(0),
            scratch_btc: floatx_slices.remove(0),
            memory_floatx: act_memory_floatx,
            memory_float: act_memory_f32,
        }
    }

    #[allow(non_snake_case)]
    fn sizes(B: usize, T: usize, C: usize, L: usize, NH: usize, Vp: usize, recompute: i32) -> ActivationTensorsSizes {
        ActivationTensorsSizes {
            encoded: B * T * C,
            ln1: L * B * T * C,
            ln1_mean: L * B * T,
            ln1_rstd: L * B * T,
            atty: L * B * T * C,
            att: L * B * NH * T * T,
            residual2: L * B * T * C,
            ln2: L * B * T * C,
            ln2_mean: L * B * T,
            ln2_rstd: L * B * T,
            fch: L * B * T * (4 * C),
            // if recompute >= 1 then we will recompute gelu_forward during backward and use this as scratch buffer
            fch_gelu: if recompute < 1 { L * B * T * (4 * C) } else { B * T * 4*C },
            residual3: L * B * T * C,
            lnf: B * T * C,
            lnf_mean: B * T,
            lnf_rstd: B * T,
            losses: B * T,
            qkvr: L * B * T * (3 * C),
            output: B * T * max(3 * C, max(NH * T, Vp)),
            scratch_bt4c: B * T * (4 * C),
            scratch_btc: B * T * C,
        }
    }
}


struct ActivationTensorsSizes {
    encoded: usize,
    ln1: usize,
    ln1_mean: usize,
    ln1_rstd: usize,
    atty: usize,
    att: usize,
    residual2: usize,
    ln2: usize,
    ln2_mean: usize,
    ln2_rstd: usize,
    fch: usize,
    fch_gelu: usize,
    residual3: usize,
    lnf: usize,
    lnf_mean: usize,
    lnf_rstd: usize,
    losses: usize,
    qkvr: usize,
    output: usize,
    scratch_bt4c: usize,
    scratch_btc: usize,
}

fn buffer_to_slices<T: DeviceCopy>(allocation: &[usize], memory: &mut DeviceBuffer<T>) -> Vec<DeviceSlice<T>> {
    let mut offset = 0usize;
    let mut out = Vec::with_capacity(allocation.len());
    for &size in allocation {
        let start = offset;
        let end = start + size;
        out.push(memory.index(start..end));
        offset = end;
    }

    out
}

struct ShardInfo {
    offset: usize,
    size: usize
}

struct GPT2 {
    config: GPT2Config,

    // the weights of the model, and their sizes
    params: ParameterTensors,
    //param_sizes: [usize; NUM_PARAMETER_TENSORS],
    //params_memory: DeviceBuffer<FloatX>,
    num_parameters: usize,

    // gradients of the weights
    grads: Option<ParameterTensors>,
    //grads_memory: Option<DeviceBuffer<FloatX>>,
    // buffers for the AdamW optimizer
    m_memory: Option<DeviceBuffer<f32>>,
    v_memory: Option<DeviceBuffer<f32>>,
    master_weights: Option<DeviceBuffer<f32>>, // is NULL unless fp32 weights is enabled.
    // the activations of the model, and their sizes
    acts: Option<ActivationTensors>,
    //act_sizes: [usize; NUM_ACTIVATION_TENSORS],
    //acts_memory_floatx: Option<DeviceBuffer<FloatX>>,
    //acts_memory_float: Option<DeviceBuffer<f32>>,
    
    batch_size: usize, // the batch size (B) of current forward pass
    seq_len: usize, // the sequence length (T) of current forward pass
    inputs: Option<DeviceBuffer<i32>>, // the input tokens for the current forward pass
    targets: Option<DeviceBuffer<i32>>, // the target tokens for the current forward pass
    mean_loss: f32,  // after the last backward micro-batch, will be populated with mean loss across all GPUs and micro-steps

    accumulated_mean_loss: Option<DeviceVariable<f32>>,  // GPU buffer used to accumulate loss across micro-steps
    cpu_losses: Option<LockedBuffer<f32>>, // CPU buffer to copy the losses to, allocated with cudaMallocHost
    rng_state: u64, // the RNG state for seeding stochastic rounding etc.
    rng_state_last_update: u64, // RNG before last gpt2_update() to re-round identically from master weights
    use_master_weights: bool, // keep master weights copy in float for optim update?
    init_state: bool,   // set to true if master weights need to be initialized
    gelu_fusion: i32, // fuse gelu via cuBLASLt (0=none, 1=forward, 2=forward+backward)
    recompute: i32, // recompute gelu | layernorm forward during model backward? 0|1|2
    // CPU scratch buffers for encoder backward
    workload_indices: Option<Vec<i32>>, // encoder_backward, B*T*num_c_groups (int)
    bucket_info: Option<Vec<Int4>>,     // encoder_backward, B*T*num_c_groups (int4) - size for worst case
}

impl GPT2 {


    #[allow(non_snake_case)]
    fn allocate_state(&mut self, B: usize, T: usize) {
        assert!(self.grads.is_none());

        self.grads = Some(ParameterTensors::allocate(&self.config));
        println!("allocated {} MiB for parameter gradients", self.grads.as_ref().unwrap().num_parameters * size_of::<FloatX>() / (1024 * 1024));

        self.batch_size = B;
        self.seq_len = T;

        self.acts = Some(ActivationTensors::allocate(&self.config, B, T, self.recompute));
        println!("allocated {} MiB for FloatX activations", self.acts.as_ref().unwrap().memory_floatx.len() * std::mem::size_of::<FloatX>() / (1024 * 1024));
        println!("allocated {} MiB for f32 activations", self.acts.as_ref().unwrap().memory_float.len() * std::mem::size_of::<f32>() / (1024 * 1024));

        // also create memory for caching inputs and targets
        self.inputs = Some(DeviceBuffer::<i32>::zeroed(B * T).unwrap());
        self.targets = Some(DeviceBuffer::<i32>::zeroed(B * T).unwrap());
        self.accumulated_mean_loss = Some(DeviceVariable::<f32>::new(0.0).unwrap());
        self.cpu_losses = Some(LockedBuffer::<f32>::new(&0.0f32,B * T).unwrap());

        // initialise cpu scratch buffers for encoder backward
        const WARP_SIZE: usize = 32;
        // X128_SIZE is the number of FloatX elements that fit in 128 bits
        const X128_SIZE: usize = 128 / (std::mem::size_of::<FloatX>() * 8);
        let num_c_groups = self.config.channels.div_ceil(WARP_SIZE * X128_SIZE);
        
        // Assert that the allocation size doesn't exceed 2^31 (todo - maybe an issue for llama3-400B(?))
        assert!((self.batch_size * self.seq_len * num_c_groups) < (1usize << 31));
        
        // Allocate workload_indices buffer: B*T*num_c_groups (int)
        self.workload_indices = Some(vec![0i32; self.batch_size * self.seq_len * num_c_groups]);
        
        // Allocate bucket_info buffer: B*T*num_c_groups (int4) - size for worst case
        self.bucket_info = Some(vec![Int4::default(); self.batch_size * self.seq_len * num_c_groups]);


        // we will now init the optimizer states and master weights
        // this is usually a substantial amount of memory allocation right here.
        let shard_num_parameters = self.num_parameters;
        println!("allocating {} MiB for AdamW optimizer state m", shard_num_parameters * size_of::<f32>() >> 20);
        println!("allocating {} MiB for AdamW optimizer state v", shard_num_parameters * size_of::<f32>() >> 20);
        assert!(self.m_memory.is_none());
        assert!(self.v_memory.is_none());

        self.m_memory = Some(DeviceBuffer::<f32>::zeroed(shard_num_parameters).unwrap());
        self.v_memory = Some(DeviceBuffer::<f32>::zeroed(shard_num_parameters).unwrap());

        if self.use_master_weights {
            assert!(self.master_weights.is_none());
            println!("allocating {} MiB for master copy of params", shard_num_parameters * size_of::<f32>() >> 20);
            self.master_weights = Some(DeviceBuffer::<f32>::zeroed(shard_num_parameters).unwrap());
        }

        // report on device memory usage
        let (free, total) = cust::memory::mem_get_info().unwrap();
        println!("device memory usage: {} MiB / {} MiB\n", (total-free) / 1024 / 1024, total / 1024 / 1024);

        // give an estimate of the maximum batch size
        let total_activation_memory = self.acts.as_ref().unwrap().memory_floatx.len() * std::mem::size_of::<FloatX>() + 
                                        self.acts.as_ref().unwrap().memory_float.len() * std::mem::size_of::<f32>();
        let bytes_per_sequence = total_activation_memory / self.batch_size;
        
        println!("memory per sequence: {} MiB", bytes_per_sequence / 1024 / 1024);
        println!(" -> estimated maximum batch size: {}", self.batch_size + free / bytes_per_sequence);

    }

    pub fn write_checkpoint(&self, checkpoint_path: &str) {
        // write the model to a checkpoint file
        println!("Writing model to {}", checkpoint_path);
        let mut model_file = File::create(checkpoint_path)
            .unwrap_or_else(|_| panic!("Error: cannot create file {:?}", checkpoint_path));
        
        // write the header first
        let mut model_header = [0u32; 256];
        model_header[0] = 20240326; // magic number
        assert!(PRECISION_MODE == PrecisionMode::Fp32 || PRECISION_MODE == PrecisionMode::Bf16);
        model_header[1] = if PRECISION_MODE == PrecisionMode::Fp32 { 3 } else { 5 }; // version
        model_header[2] = self.config.max_seq_len as u32;
        model_header[3] = self.config.vocab_size as u32;
        model_header[4] = self.config.num_layers as u32;
        model_header[5] = self.config.num_heads as u32;
        model_header[6] = self.config.channels as u32;
        model_header[7] = self.config.padded_vocab_size as u32;
        
        // Write header as little-endian u32 array
        for &value in &model_header {
            model_file.write_all(&value.to_le_bytes()).unwrap();
        }
        
        // write the parameters
        let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();
        let buf_size = 32 * 1024 * 1024 / std::mem::size_of::<f32>(); // IO_BUF_SIZE
        llm_rs::cuda_utils::device_to_file(&mut model_file, &self.params.memory, buf_size, &stream);
        
        // close file, we're done
        drop(model_file);
    }

    /// Build GPT2 model from a checkpoint file
    pub fn from_checkpoint(checkpoint_path: &str, weight_init: bool, stream: &Stream) -> Self {
        // If weight_init is true, we will load the weights from this checkpoint .bin file
        // We sometimes want this to be false, if we are going to initialize these weights from
        // the master weights that are instead stored in the state .bin file.
        // In that case, this function mostly loads the model hyperparameters from the header.

        if PRECISION_MODE == PrecisionMode::Fp16 {
             panic!("build_from_checkpoint() does not support fp16 right now.");
        }

        // read in model from a checkpoint file
        let model_file = File::open(checkpoint_path)
            .unwrap_or_else(|_| panic!("Error: cannot open file {:?}", checkpoint_path));
        let mut model_file_reader = BufReader::new(model_file);
        let model_header: [u32; 256] = read_le_u32_array::<_, 256>(&mut model_file_reader);
        if model_header[0] != 20240326 { panic!("Bad magic model file"); }
        
        let version = model_header[1];
        if !(version == 3 || version == 5) {
            // 3 = fp32, padded vocab
            // 5 = bf16, padded vocab, layernorms also in bf16
            eprintln!("Bad version in model file");
            eprintln!("---> HINT: try to re-run `python train_gpt2.py`");
            panic!();
        }

        // check if the precision mode of the checkpoint matches the model precision
        if weight_init {
            if PRECISION_MODE == PrecisionMode::Bf16 && version != 5 {
                eprintln!("Precision is configured as BF16 but model at {} is not.", checkpoint_path);
                eprintln!("---> HINT: are you sure you're loading a _bf16.bin file?");
                panic!();
            }
            if PRECISION_MODE == PrecisionMode::Fp32 && version != 3 {
                eprintln!("Precision is configured as FP32 but model at {} is not.", checkpoint_path);
                eprintln!("---> HINT: to turn on FP32 you have to compile like: `make train_gpt2cu PRECISION=FP32`");
                eprintln!("---> HINT: are you sure you're loading a .bin file without any _bf16 in the name?");
                panic!();
            }
        }

        let config = GPT2Config {
            max_seq_len: model_header[2] as usize,
            vocab_size: model_header[3] as usize,
            num_layers: model_header[4] as usize,
            num_heads: model_header[5] as usize,
            channels: model_header[6] as usize,
            padded_vocab_size: model_header[7] as usize
        };

        // allocate memory for the model parameters
        let mut params = ParameterTensors::allocate(&config);
        println!("allocated {} MiB for model parameters", params.num_parameters * std::mem::size_of::<FloatX>() / (1024 * 1024));
        let num_parameters = params.num_parameters;

        // read in the parameters if weight_init is true
        if weight_init {
            llm_rs::cuda_utils::file_to_device(&mut params.memory, &mut model_file_reader, 1024, &stream);
        }

        // only return from this function once we are certain the params are ready on the GPU
        stream.synchronize().unwrap();

        Self {
            config: config,
            params: params,
            grads: None,
            m_memory: None,
            v_memory: None,
            master_weights: None,
            num_parameters: num_parameters,
            acts: None,
            batch_size: 0,
            seq_len: 0,
            inputs: None,
            targets: None,
            mean_loss: -1.0, // -1.0 designates no loss
            accumulated_mean_loss: None,
            cpu_losses: None,
            rng_state: 13371337, // used in stochastic rounding  // TODO + multi_gpu_config.process_rank;
            rng_state_last_update: 0,
            use_master_weights: true, // safe default: do keep master weights in fp32
            init_state: true,
            gelu_fusion: 0, // default: off for now
            recompute: 1, // good default: recompute gelu but not layernorm
            workload_indices: None,
            bucket_info: None,
        }
    }

    fn parse_gpt2_hyperparameters(depth_str: &str) -> Result<GPT2Config, String> {
        let depth: usize = depth_str.parse().map_err(|_| "Invalid depth (not a number)".to_string())?;

        let (channels, num_heads) = match depth {
            6  => (384, 6),     // (unofficial) gpt2-tiny (30M)
            12 => (768, 12),    // gpt2 (124M)
            24 => (1024, 16),   // gpt2-medium (350M)
            36 => (1280, 20),   // gpt2-large (774M)
            48 => (1600, 25),   // gpt2-xl (1558M)
            60 => (1920, 30),   // (unofficial) 2.7B
            72 => (2880, 30),   // (unofficial) 7.3B
            84 => (3456, 36),   // (unofficial) 12.2B
            _  => return Err(format!("Unsupported GPT-2 depth: {}", depth)),
        };
        
        Ok(GPT2Config {
            vocab_size: VOCAB_SIZE,
            padded_vocab_size: PADDED_VOCAB_SIZE,
            num_layers: depth,
            channels: channels,
            num_heads: num_heads,
            max_seq_len: 1024,
        })
    }

    fn parse_gpt3_hyperparameters(channels_str: &str) -> Result<GPT2Config, String> {
        let channels: usize = channels_str.parse().map_err(|_| "invalid GPT-3 channels".to_string())?;

        let (depth, head_size) = match channels {
            384    => (6,   64),  // (unofficial) gpt3-tiny (31M)
            768    => (12,  64),  // gpt3-small (125M)
            1024   => (24,  64),  // gpt3-medium (350M)
            1536   => (24,  96),  // gpt3-large (760M)
            2048   => (24, 128),  // gpt3-xl (1.3B) [heads fixed]
            2560   => (32,  80),  // gpt3-2.7B
            4096   => (32, 128),  // gpt3-6.7B
            5140   => (40, 128),  // gpt3-13B
            12288  => (96, 128),  // gpt3 (175B)
            _ => return Err(format!("unsupported GPT-3 channels: {}", channels)),
        };
        if channels % head_size != 0 {
            return Err(format!("channels {} not divisible by head size {}", channels, head_size));
        }
        
        Ok(GPT2Config {
            vocab_size: VOCAB_SIZE,
            padded_vocab_size: PADDED_VOCAB_SIZE,
            num_layers: depth,
            channels: channels,
            num_heads: channels / head_size,
            max_seq_len: 2048, // NOTE: GPT-3 uses context length of 2048 tokens, up from 1024 in GPT-2
        })
    }

    /// Build GPT2 model from a descriptor string
    #[allow(non_snake_case)]
    pub fn from_descriptor(descriptor: &str) -> Self {
        // The model descriptor can be:
        // - legacy format "dX", where X is number, e.g. "d12". This creates GPT-2 model with 12 layers.
        // - new explicit format "gpt2:dX", same as above, e.g. "gpt2:d48" for GPT-2 with 48 layers.
        // - "gpt3:cX", where X is now the channel count, e.g. "gpt3:c768" is the smallest GPT-3 model.
        
        let config = if let Some(descriptor_suffix) = descriptor.strip_prefix('d') {
             GPT2::parse_gpt2_hyperparameters(descriptor_suffix).unwrap()
        } else if let Some(descriptor_suffix) = descriptor.strip_prefix("gpt2:d") {
            GPT2::parse_gpt2_hyperparameters(descriptor_suffix).unwrap()
        } else if let Some(descriptor_suffix) = descriptor.strip_prefix("gpt3:c") {
            GPT2::parse_gpt3_hyperparameters(descriptor_suffix).unwrap()
        } else {
            panic!("Unsupported model descriptor: {}", descriptor);
        };

        let mut params = ParameterTensors::allocate(&config);
        let num_params = params.num_parameters;
        let param_sizes = params.sizes.to_vec();

        // allocate and random init the memory for all the parameters with GPT-2 schema
        // weights ~N(0, 0.02), biases 0, c_proj weights ~N(0, 0.02/(2*L)**0.5)
        // NOTE: assuming all parameters are of the type floatX, could be relaxed later
        use llm_rs::random::{Mt19937, normal_};
        
        let mut init_rng = Mt19937::new(42);
        let mut params_memory_cpu = vec![zero_floatx(); num_params];
        
        // fill in all the weights with random values
        let residual_scale = 1.0f32 / (2.0f32 * config.num_layers as f32).sqrt();
        
        // we have to init all these tensors exactly in the order that PyTorch initializes them
        // so that we can match them up and get correctness and exactly the same initial conditions
        let L = config.num_layers;
        let mut offset ;
        
        for l in 0..L {
            offset = 0;
            for i in 0..NUM_PARAMETER_TENSORS {
                // the layernorm parameters are all initialized to 1
                if l == 0 && (i == 2 || i == 8 || i == 14) { // only at l = 0 to init these just once
                    let ones = vec_f32_to_floatx(&vec![1.0f32; param_sizes[i]]);
                    params_memory_cpu[offset..offset + param_sizes[i]].copy_from_slice(&ones);
                }
                // weights tensors are handled here
                if (l == 0 && (i == 0 || i == 1)) // only at l = 0, init the wte and wpe tensors
                  || i == 4 || i == 6 || i == 10 || i == 12 {
                    let mut n = param_sizes[i];
                    let mut layer_offset = 0;
                    if i == 0 {
                        // for wte tensor (padded vocab) override to init V instead of Vp rows
                        n = config.vocab_size * config.channels;
                    }
                    if i == 4 || i == 6 || i == 10 || i == 12 {
                        // weight tensors, we are only initializing layer l
                        assert!(n % L == 0);
                        n = n / L;
                        layer_offset = l * n;
                    }
                    // in GPT-2, the projections back into the residual stream are additionally
                    // scaled by 1/sqrt(2*L) for training stability
                    let scale = if i == 6 || i == 12 { 0.02f32 * residual_scale } else { 0.02f32 };
                    // okay let's draw the random numbers and write them
                    let mut fp32_buffer = vec![0.0f32; n];
                    normal_(&mut fp32_buffer, 0.0f32, scale, &mut init_rng);
                    let floatx_buffer = vec_f32_to_floatx(&fp32_buffer);
                    params_memory_cpu[offset + layer_offset ..offset + layer_offset  + n].copy_from_slice(&floatx_buffer);
                }
                offset += param_sizes[i];
            }
        }
        
        // copy them to GPU
        params.memory.copy_from(&params_memory_cpu).unwrap();
        
        Self {
            config: config,
            params: params,
            grads: None,
            m_memory: None,
            v_memory: None,
            master_weights: None,
            num_parameters: num_params,
            acts: None,
            batch_size: 0,
            seq_len: 0,
            inputs: None,
            targets: None,
            mean_loss: -1.0,
            accumulated_mean_loss: None,
            cpu_losses: None,
            rng_state: 13371337,
            rng_state_last_update: 0,
            use_master_weights: true,
            init_state: true,
            gelu_fusion: 0,
            recompute: 1,
            workload_indices: None,
            bucket_info: None,
        }
    }

    fn check_tokens(tokens: &[i32], vocab_size: usize) {
        tokens.iter().enumerate().for_each(| (i, t) | if *t < 0 || *t >= vocab_size as i32 {
            panic!("Error: Token out of vocabulary {} position {}", t, i);
        })

    }

    fn get_layer_params<T: DeviceCopy>(memory: DeviceSlice<T>, layer: usize, layer_size: usize) -> DeviceSlice<T> {
        let offset = layer * layer_size;
        memory.index(offset..offset + layer_size)
    }


    #[allow(non_snake_case)]
    pub fn forward(&mut self, inputs: &[i32], B: usize, T: usize, stream: &Stream) {
        // NVTX_RANGE_FN();
        // we must be careful and use size_t instead of int, otherwise
        // we could overflow int. E.g. l * B * NH * T * T overflows int at B 16.

        // convenience parameters
        let V = self.config.vocab_size;
        let Vp = self.config.padded_vocab_size;
        let L = self.config.num_layers;
        let NH = self.config.num_heads;
        let C = self.config.channels;

        if B > self.batch_size || T > self.seq_len {
            panic!("Model: B={} T={}, Desired: B={} T={}", self.batch_size, self.seq_len, B, T);
        }

        self.inputs.as_mut().unwrap().copy_from(inputs).unwrap();
        // validate inputs, all indices must be in the range [0, V)
        // we can do this while the copies are already underway
        GPT2::check_tokens(inputs, V);

        // forward pass
        let params = &self.params;
        let acts = self.acts.as_mut().unwrap();
        k_encoder_forward(&mut acts.encoded, &self.inputs.as_ref().unwrap(), &params.wte, &params.wpe, B, T, C, stream);
        
        // first layernorm isn't fused
        let mut ln1 = if self.recompute < 2 { acts.ln1 } else { acts.lnf };
        k_layernorm_forward(&mut ln1 , &mut acts.ln1_mean, &mut acts.ln1_rstd, &acts.encoded, &params.ln1w, &params.ln1b, B, T, C, stream);

        let mut pre_gelu_default = DevicePointer::<FloatX>::null();
        let gelu_fusion_default = 0;

        for l in 0..L {
            // NvtxRange layer_range("Layer", l);

            let residual = if l == 0 { acts.encoded } else { GPT2::get_layer_params(acts.residual3, l-1, B * T * C) };

            // get the pointers of the weights for this layer
            let l_qkvw = GPT2::get_layer_params(params.qkvw, l,  3*C * C);
            let l_qkvb = GPT2::get_layer_params(params.qkvb, l, 3*C);
            let l_attprojw = GPT2::get_layer_params(params.attprojw, l, C * C);
            let l_attprojb = GPT2::get_layer_params(params.attprojb, l, C);
            let l_ln2w = GPT2::get_layer_params(params.ln2w, l, C);
            let l_ln2b = GPT2::get_layer_params(params.ln2b, l, C);
            let l_fcw = GPT2::get_layer_params(params.fcw, l, 4*C * C);
            let l_fcb = GPT2::get_layer_params(params.fcb, l, 4*C);
            let l_fcprojw = GPT2::get_layer_params(params.fcprojw, l, C * 4*C);
            let l_fcprojb = GPT2::get_layer_params(params.fcprojb, l, C);

            // get the pointers of the activations for this layer
            let l_ln1 = if self.recompute < 2 { GPT2::get_layer_params(acts.ln1, l, B * T * C) } else { acts.lnf };
            let mut l_qkvr = GPT2::get_layer_params(acts.qkvr, l, B * T * 3*C);
            let mut l_atty = GPT2::get_layer_params(acts.atty, l, B * T * C);
            let mut l_residual2 = GPT2::get_layer_params(acts.residual2, l, B * T * C);
            let mut l_ln2 = if self.recompute < 2 { GPT2::get_layer_params(acts.ln2, l, B * T * C) } else { acts.lnf };
            let mut l_ln2_mean = GPT2::get_layer_params(acts.ln2_mean, l, B * T);
            let mut l_ln2_rstd = GPT2::get_layer_params(acts.ln2_rstd, l, B * T);
            let l_fch = GPT2::get_layer_params(acts.fch, l, B * T * 4*C);
            // reuse the same activation buffer at each layer, as we'll re-compute the gelu during backward
            // very useful because we dramatically reduce VRAM usage, and may be able to fit larger batch size
            let mut l_fch_gelu = if self.recompute < 1 { GPT2::get_layer_params(acts.fch_gelu, l, B * T * 4*C) } else { acts.fch_gelu };
            let mut l_residual3 = GPT2::get_layer_params(acts.residual3, l, B * T * C);
            let mut scratch = acts.output;  // used for non-cudnn attention, fcproj, attproj, etc.

            let mut l_att = GPT2::get_layer_params(acts.att, l, B * NH * T * T);

            // unused parts of attention buffer must be zeroed (T-dependent)
            if T != self.seq_len {
                l_att.set_zero().unwrap();
            }

            // these are only needed as scratchpads for the forward pass, but
            // need not be stored for backward
            k_matmul_forward_cublaslt(&mut scratch, &l_ln1, &l_qkvw, &l_qkvb.as_device_ptr(), B, T, C, 3*C, stream, &mut pre_gelu_default, gelu_fusion_default);
            k_attention_forward(&mut l_atty, &mut l_qkvr, &mut l_att, &scratch, B, T, C, NH, stream);
            
            k_matmul_forward_cublaslt(&mut scratch, &l_atty, &l_attprojw, &l_attprojb.as_device_ptr(), B, T, C, C, stream, &mut pre_gelu_default, gelu_fusion_default);
            k_fused_residual_forward5(&mut l_residual2, &mut l_ln2, &mut l_ln2_mean, &mut l_ln2_rstd, &residual, &scratch, &l_ln2w, &l_ln2b, B*T, C, stream);
            k_matmul_forward_cublaslt(&mut l_fch_gelu, &l_ln2, &l_fcw, &l_fcb.as_device_ptr(), B, T, C, 4*C, stream, &mut l_fch.as_device_ptr(), self.gelu_fusion);
            k_matmul_forward_cublaslt(&mut scratch, &l_fch_gelu, &l_fcprojw, &l_fcprojb.as_device_ptr(), B, T, 4*C, C, stream, &mut pre_gelu_default, gelu_fusion_default);
            

            // OK, fusion across blocks.
            if l+1 != L {
                let mut l_ln1 = if self.recompute < 2 { acts.ln1.index((l+1) * B * T * C .. (l+2) * B * T * C) } else { acts.lnf };
                let mut l_ln1_mean = acts.ln1_mean.index((l+1) * B * T .. (l+2) * B * T);
                let mut l_ln1_rstd = acts.ln1_rstd.index((l+1) * B * T .. (l+2) * B * T);
                let l_ln1w = params.ln1w.index((l+1) * C .. (l+2) * C);
                let l_ln1b = params.ln1b.index((l+1) * C .. (l+2) * C);
                k_fused_residual_forward5(&mut l_residual3, &mut l_ln1, &mut l_ln1_mean, &mut l_ln1_rstd, 
                    &l_residual2, &scratch, &l_ln1w, &l_ln1b, 
                    B*T, C, stream);
            } else {
                k_fused_residual_forward5(&mut l_residual3, &mut acts.lnf, &mut acts.lnf_mean, &mut acts.lnf_rstd, 
                    &l_residual2, &scratch, &params.lnfw, &params.lnfb,
                    B*T, C, stream);
            }
        }

        k_matmul_forward_cublaslt(&mut acts.output, & acts.lnf, &params.wte, &DevicePointer::<FloatX>::null(), B, T, C, Vp, stream, &mut pre_gelu_default, gelu_fusion_default);
        stream.synchronize().unwrap();
    }

    // Forwards both the model and the loss and is used for validation splits and evals.
    // In particular it populates cpu_losses with loss at each token.
    // Some of the evals (e.g. HellaSwag) require the per-token losses, which are produced here.
    #[allow(non_snake_case)]
    pub fn validate(&mut self, inputs: &[i32], targets: &[i32], B: usize, T: usize, stream: &Stream) -> f32 {
        // forward the model itself
        self.forward(inputs, B, T, stream);

        let V = self.config.vocab_size;
        let Vp = self.config.padded_vocab_size;

        // NvtxRange classifier_and_loss_range("classifier_and_loss");
        let acts = self.acts.as_mut().unwrap();
        let mut mean_loss = 0.0f32;
        // fused classifier: does the forward pass and first part of the backward pass
        let dloss = 1.0f32 / (B * T) as f32; // results in the uniform average loss over all elements
        // note: we don't need to generate dlogits here
        acts.losses.set_zero().unwrap();
        self.targets.as_mut().unwrap().copy_from(targets).unwrap();
        GPT2::check_tokens(targets, V);
        k_fused_classifier(&mut acts.output, &mut acts.losses, dloss, &self.targets.as_ref().unwrap(),
             B, T, V, Vp, false, stream);
        let cpu_losses = self.cpu_losses.as_deref_mut().unwrap();
        acts.losses.copy_to(cpu_losses).unwrap();
        for i in 0..B*T {
            mean_loss += cpu_losses[i];
        }
        mean_loss /= (B*T) as f32;
        stream.synchronize().unwrap();

        mean_loss
    }

    #[allow(non_snake_case)]
    pub fn backward_and_reduce(&mut self, inputs: &[i32], targets: &[i32], grad_accum_steps: usize, micro_step: usize, stream: &Stream) {
        if self.grads.is_none() {
            panic!("Need to allocate gradients before backward");
        }

        //NVTX_RANGE_FN();
        let params = &mut self.params;
        let grads = self.grads.as_mut().unwrap();
        let acts = self.acts.as_mut().unwrap();
        let d_targets = self.targets.as_mut().unwrap();

        let last_step = micro_step == grad_accum_steps - 1;
        // on the first micro-step zero the gradients, as we're about to += accumulate into them
        if micro_step == 0 {
            // there are currently two state vars during the gradient accumulation inner loop:
            // 1) the losses accumulate += into acts.losses, reset here
            // 2) the gradients accumulate += into grads_memory, reset here
            acts.losses.set_zero().unwrap();
            grads.memory.set_zero().unwrap();
        }

        // convenience shortcuts, size_t instead of int so that pointer arithmetics don't overflow
        let B = self.batch_size;
        let T = self.seq_len;
        let V = self.config.vocab_size;
        let Vp = self.config.padded_vocab_size;
        let L = self.config.num_layers;
        let NH = self.config.num_heads;
        let C = self.config.channels;

        // accumulate the losses inside acts.losses, and kick off the backward pass inside the fused classifier
        //NvtxRange classifier_and_loss_range("classifier_and_loss");
        let dloss = 1.0f32 / (B * T * grad_accum_steps) as f32; // results in the uniform average loss over all elements
        d_targets.copy_from(targets).unwrap();
        GPT2::check_tokens(targets, V);
        //debug::debug_print_device_tensor(acts.output, "[backward] logits=acts.output", stream);
        k_fused_classifier(&mut acts.output, &mut acts.losses, dloss, &d_targets, B, T, V, Vp, true, stream);
        
        // backward pass: go in the reverse order of the forward pass, and call backward() functions

        // reset residual stream gradients (put here to work with gradient accumulation)
        let mut dresidual = acts.scratch_btc; // the main buffer holding the gradient in the backward pass
        dresidual.set_zero().unwrap();

        // borrow acts.output and call it scratchF
        let scratchF = &mut borrow_as_f32_mut(&mut acts.output);
        let mut device_floatx_none: Option<DeviceSlice<FloatX>> = None;
        let device_f32_none: Option<DeviceSlice<f32>> = None;
        let mut default_pre_gelu = device_floatx_none;
        let default_gelu_fusion = 1;
        
        // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
        // this was done in the fused classifier kernel as last step of forward pass
        // technically that is a small, inline backward() pass of calculating
        // total, final loss as the mean over all losses over all (B,T) positions in the batch
        // next: backward the classifier matmul
        k_matmul_backward(&mut acts.scratch_bt4c, &mut grads.wte, &mut device_floatx_none, &acts.output, &acts.lnf, &params.wte, device_f32_none, B, T, C, Vp, stream, &mut default_pre_gelu, default_gelu_fusion);
        // backward the final layernorm
        let mut residual =  GPT2::get_layer_params(acts.residual3, L-1,  B * T * C);
        k_layernorm_backward(&mut dresidual, &mut grads.lnfw, &mut grads.lnfb, scratchF, &acts.scratch_bt4c, &residual, &params.lnfw, &acts.lnf_mean, &acts.lnf_rstd, B, T, C, stream);

        // from this point on, we no longer need the values stored in the last residual, so we can reuse that memory as generic
        // scratch for backward computations
        let mut dl_btc = residual;

        // now backward all the layers
        for l in (0..L).rev() {
            //NvtxRange layer_range("Layer", l);

            residual = if l == 0 { acts.encoded } else { GPT2::get_layer_params(acts.residual3, l-1, B*T*C) };
            let scratchF = &mut borrow_as_f32_mut(&mut acts.output);

            // get the pointers of the weights for this layer
            let l_ln1w = GPT2::get_layer_params(params.ln1w, l, C);
            let l_ln1b = GPT2::get_layer_params(params.ln1b, l, C);
            let l_qkvw = GPT2::get_layer_params(params.qkvw, l, 3*C * C);
            let l_attprojw = GPT2::get_layer_params(params.attprojw, l, C * C);
            let l_ln2w = GPT2::get_layer_params(params.ln2w, l, C);
            let l_ln2b = GPT2::get_layer_params(params.ln2b, l, C);
            let l_fcw = GPT2::get_layer_params(params.fcw, l, 4*C * C);
            let l_fcprojw = GPT2::get_layer_params(params.fcprojw, l, C * 4*C);
            // get the pointers of the gradients of the weights for this layer
            let mut dl_ln1w = GPT2::get_layer_params(grads.ln1w, l, C);
            let mut dl_ln1b = GPT2::get_layer_params(grads.ln1b, l, C);
            let mut dl_qkvw = GPT2::get_layer_params(grads.qkvw, l, 3*C * C);
            let dl_qkvb = GPT2::get_layer_params(grads.qkvb, l, 3*C);
            let mut dl_attprojw = GPT2::get_layer_params(grads.attprojw, l, C * C);
            let dl_attprojb = GPT2::get_layer_params(grads.attprojb, l, C);
            let mut dl_ln2w = GPT2::get_layer_params(grads.ln2w, l, C);
            let mut dl_ln2b = GPT2::get_layer_params(grads.ln2b, l, C);
            let mut dl_fcw = GPT2::get_layer_params(grads.fcw, l, 4*C * C);
            let dl_fcb = GPT2::get_layer_params(grads.fcb, l, 4*C);
            let mut dl_fcprojw = GPT2::get_layer_params(grads.fcprojw, l, C * 4*C);
            let dl_fcprojb = GPT2::get_layer_params(grads.fcprojb, l, C);
            // get the pointers of the activations for this layer
            let mut l_ln1 = if self.recompute < 2 { GPT2::get_layer_params(acts.ln1, l, B * T * C) } else { acts.lnf };
            let mut l_ln1_mean = GPT2::get_layer_params(acts.ln1_mean, l, B * T);
            let mut l_ln1_rstd = GPT2::get_layer_params(acts.ln1_rstd, l, B * T);
            let l_qkvr = GPT2::get_layer_params(acts.qkvr, l, B * T * 3*C);
            let l_atty = GPT2::get_layer_params(acts.atty, l, B * T * C);
            let l_residual2 = GPT2::get_layer_params(acts.residual2, l, B * T * C);
            let mut l_ln2 = if self.recompute < 2 { GPT2::get_layer_params(acts.ln2, l, B * T * C) } else { acts.lnf };
            let mut l_ln2_mean = GPT2::get_layer_params(acts.ln2_mean, l, B * T);
            let mut l_ln2_rstd = GPT2::get_layer_params(acts.ln2_rstd, l, B * T);
            let l_fch_pre_gelu = GPT2::get_layer_params(acts.fch, l, B * T * 4*C);
            let mut l_fch_gelu = if self.recompute < 1 { GPT2::get_layer_params(acts.fch_gelu, l, B * T * 4*C) } else { acts.fch_gelu };
            // get the pointers of the gradients of the activations for this layer
            // notice that there is no l *, because we just have a single copy, and keep
            // re-using this memory in every Transformer block as we calculate backward pass
            let mut dl_bt4c = acts.scratch_bt4c;

            if self.recompute >= 1 {
                // recompute >= 1 means we recompute gelu. in this case,
                // l_fch_gelu is just a buffer, so re-compute the gelu from l_fch here
                k_gelu_forward(&mut l_fch_gelu, &l_fch_pre_gelu, B*T*4*C, stream);
            }
            k_matmul_backward(&mut dl_bt4c, &mut dl_fcprojw, &mut Some(dl_fcprojb), &dresidual, &l_fch_gelu, &l_fcprojw, Some(*scratchF), B, T , 4*C, C, stream, &mut Some(l_fch_pre_gelu), self.gelu_fusion);
            if self.recompute >= 2 {
                // same as gelu above, l_ln1 and l_ln2 are just buffers if recompute >= 2, recompute them here on demand
                k_layernorm_forward(&mut l_ln2, &mut l_ln2_mean, &mut l_ln2_rstd, &l_residual2, &l_ln2w, &l_ln2b, B, T, C, stream);
            }
            k_matmul_backward(&mut dl_btc, &mut dl_fcw, &mut Some(dl_fcb), &dl_bt4c, &l_ln2, &l_fcw, Some(*scratchF), B, T, C, 4 * C, stream, &mut default_pre_gelu, default_gelu_fusion);
            // layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
            k_layernorm_backward(&mut dresidual, &mut dl_ln2w, &mut dl_ln2b, scratchF, &dl_btc, &l_residual2, &l_ln2w, &l_ln2_mean, &l_ln2_rstd, B, T, C, stream);
            k_matmul_backward(&mut dl_btc, &mut dl_attprojw, &mut Some(dl_attprojb), &dresidual, &l_atty, &l_attprojw, Some(*scratchF), B, T, C, C, stream, &mut default_pre_gelu, default_gelu_fusion);
            let l_att = GPT2::get_layer_params(acts.att, l,  B * NH * T * T);
            // we need B x T x (4)C buffers. l_atty and l_fch aren't needed anymore at this point, so reuse their memory
            
            let _ = scratchF; // release scratchF so we can borrow acts.output as scratchX
            let mut scratchX = &mut acts.output;
            let mut buffer_a = l_atty;
            let mut buffer_b = l_fch_pre_gelu;
            k_attention_backward(&mut dl_bt4c, &mut buffer_b, &mut scratchX, &mut buffer_a, &dl_btc, &l_qkvr, &l_att, B, T, C, NH, stream);

            if self.recompute >= 2 {
                k_layernorm_forward(&mut l_ln1, &mut l_ln1_mean, &mut l_ln1_rstd, &residual, &l_ln1w, &l_ln1b, B, T, C, stream);
            }

            let _ = scratchX; // now release scratchX so we can borrow acts.output as scratchF
            let scratchF = &mut borrow_as_f32_mut(&mut acts.output);

            // QKV parameter gradients
            k_matmul_backward(&mut dl_btc, &mut dl_qkvw, &mut Some(dl_qkvb), &dl_bt4c, &l_ln1, &l_qkvw, Some(*scratchF), B, T, C, 3 * C, stream, &mut default_pre_gelu, default_gelu_fusion);
            // layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
            k_layernorm_backward(&mut dresidual, &mut dl_ln1w, &mut dl_ln1b, scratchF, &mut dl_btc, &residual, &l_ln1w, &l_ln1_mean, &l_ln1_rstd, B, T, C, stream);

            // Accumulate gradients from this layer in a background stream.
            // TODO
            //multi_gpu_async_reduce_gradient(pointers, nelem, &multi_gpu_config, main_stream);
            
        }

        let _ = scratchF; // discard scratchF
        let mut scratchX = &mut acts.output;

        k_encoder_backward(&mut grads.wte, &mut grads.wpe, &mut scratchX, &mut self.workload_indices.as_mut().unwrap(), &mut self.bucket_info.as_mut().unwrap(),
            &dresidual, self.inputs.as_ref().unwrap(), inputs, B, T, C, sampler::random_u32(&mut self.rng_state), stream);

        // Aggregate all gradients that are not part of the transformer blocks
        if last_step {
            // reduce all the losses within the current GPU (across all microsteps)
            k_global_sum_deterministic_float(&mut self.accumulated_mean_loss.as_mut().unwrap(), &acts.losses, B*T, stream);
            
            self.accumulated_mean_loss.as_mut().unwrap().copy_dtoh().unwrap();
            self.mean_loss = **self.accumulated_mean_loss.as_ref().unwrap();
            //TODO multi GPU
        }

        if last_step {
            self.mean_loss /= (B*T*grad_accum_steps) as f32;
        } else {
            self.mean_loss = -1.0f32; // no loss available yet
        }
        
    }

    pub fn calculate_grad_norm(&self, multi_gpu_config: &MultiGpuConfig, stream: &Stream) -> f32 {
        //NVTX_RANGE_FN();
        
        let grads_memory = self.grads.as_ref().unwrap().memory.as_device_ptr();
        
        let num_slices: [i32; 2] = [1, self.config.num_layers as i32];
        let max_num_block_sums = unsafe {get_max_num_block_sums(num_slices.as_ptr(), 2)};

        // FIXME keep that buffer allocated
        let grad_norm_squared = DeviceBuffer::<f32>::zeroed(max_num_block_sums as usize).unwrap();
        let mut grad_norm_squared_result = DeviceVariable::<f32>::new(0.0).unwrap();

        if multi_gpu_config.zero_stage == 1 {
            panic!("multi gpu is not supported");
        } else {
            // in regular DDP, backward has averaged the gradients across all GPUs
            // so each GPU can compute the squared norm over the whole grad vector, with no added comms needed
            k_global_norm_squared(grad_norm_squared.as_device_ptr(), grads_memory, self.num_parameters, 0, 1, max_num_block_sums, true, stream);
            k_global_sum_deterministic_float(&mut grad_norm_squared_result, &grad_norm_squared, max_num_block_sums as usize, stream);
        }

        grad_norm_squared_result.copy_dtoh().unwrap();
        let grad_norm_squared_cpu = *grad_norm_squared_result;
        grad_norm_squared_cpu.sqrt()
    }

    pub fn update(&mut self, learning_rate: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32, grad_scale: f32, t: i32, init_from_master_only: bool, stream: &Stream) {
        // update the model parameters using the AdamW optimizer
        // keep in mind that optimizer sharding (ZeRO-1) assigns different parameters to different GPUs
        // so we may not be responsible for the entire parameter tensor
        // also, this function was very simple a while back but become very complex, only because we want to
        // selectively weight decay some, but not all tensors :(
        // TODO: revisit and probably refactor this entire function
        if self.grads.is_none() || self.m_memory.is_none() || self.v_memory.is_none() {
            panic!("Need to allocate optimizer state before update");
        }

        let init_state = self.init_state;
        self.init_state = false;

        // save RNG state at this point so we can round from master weights identically when restoring from a checkpoint
        self.rng_state_last_update = self.rng_state;

        for tensor_id in 0..NUM_PARAMETER_TENSORS {
            let seed = sampler::random_u32(&mut self.rng_state);

            let mut num_layers = self.config.num_layers;
            if tensor_id < 2 || tensor_id > 13 {
                num_layers = 1;
            }

            let tensor = self.get_tensor_at_layer(0, tensor_id);
            let shard = ShardInfo{ offset: 0, size: tensor.size};
            let local_offset_full = tensor.offset + shard.offset;


            // we only want to weight decay the 2D tensors and leave all 1D tensors alone
            // in particular this also decays the embedding weights, but this is ok:
            // - the token embeddings are weight shared and participate in the final projection to logits
            // - the position embeddings actively participate at every forward/backward pass
            let wd = if tensor_id == 0 || tensor_id == 1 || tensor_id == 4 || tensor_id == 6 || tensor_id == 10 || tensor_id == 12 { weight_decay } else { 0.0f32 };
            let mut param_ptr = unsafe { self.params.memory.as_device_ptr().offset(local_offset_full as isize)};
            let grad_ptr = unsafe {self.grads.as_ref().unwrap().memory.as_device_ptr().offset(local_offset_full as isize)};

            let opt_state_offset = local_offset_full;
            let m_ptr = unsafe {self.m_memory.as_ref().unwrap().as_device_ptr().offset(opt_state_offset as isize)};
            let v_ptr = unsafe {self.v_memory.as_ref().unwrap().as_device_ptr().offset(opt_state_offset as isize)};
            let mut master_ptr = if let Some(master_weights) = &self.master_weights {
                unsafe {master_weights.as_device_ptr().offset(opt_state_offset as isize)}
            } else {
                cust::memory::DevicePointer::null()
            };

            if init_state && self.master_weights.is_some() {
                let grid_size = shard.size.div_ceil(512);
                k_copy_and_cast(&mut master_ptr, &param_ptr, shard.size, shard.size, tensor.size, grid_size, num_layers, stream);
            }
            
            if init_from_master_only {
                // when resuming training from a checkpoint with master weights (allows changing precision)
                k_init_from_master(&mut param_ptr, &master_ptr, shard.size, tensor.size, shard.size, num_layers, seed, stream);
            } else {
            // ok finally call the kernel to update the weights with AdamW
                k_adamw_update(&mut param_ptr, &master_ptr, &grad_ptr,
                    &m_ptr, &v_ptr,
                    shard.size, tensor.size, tensor.size, shard.size, num_layers,
                    learning_rate,
                    beta1, beta2, t, eps, wd, grad_scale, seed, stream);
            }            
        }

    }


    fn get_tensor_at_layer(&self, layer_id: usize, param_tensor_id: usize) -> ShardInfo {
        // first offset our way to the parameter tensor start
        let mut offset = 0;

        let param_sizes = self.params.sizes.to_vec();

        for i in 0..param_tensor_id {
            offset += param_sizes[i];
        };

        let mut size = param_sizes[param_tensor_id];
        // if we are in the transformer block, we need to additionally offset by the layer id
        if 2 <= param_tensor_id && param_tensor_id <= 13 {
            size /= self.config.num_layers;
            offset += layer_id * size;
        };

        return ShardInfo { offset: offset, size: size }
    }


    pub fn estimate_mfu(&self, _tokens_processed: usize, _time_elapsed: f32) -> f32 {
        // TODO
        -1.0f32
    }

}


fn common_start(_override_enable_tf32: bool) -> cust::error::CudaResult<Stream> {
    let device_idx = 0;
    let device = Device::get_device(device_idx)?;
    println!("Device #{}, Name: {}", device_idx, device.name()?);

    unsafe { cublas_init() };

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    Ok(stream)
}

fn save_state(filename: &str, step: i32, model: &GPT2, loader: &Dataloader, stream: &Stream) {
    println!("Writing state to {}", filename);
    let mut state_file = File::create(filename).expect("Failed to create state file");
    
    let mut state_header = [0i32; 256];
    // basic identifying information
    state_header[0] = 20240527; // magic number
    state_header[1] = 1; // version number
    state_header[2] = 1; // number of processes - TODO: multi_gpu_config.num_processes
    state_header[3] = 0; // rank of this process - TODO: multi_gpu_config.process_rank
    state_header[4] = i32::from(model.use_master_weights);  // whether we're using fp32 master weights
    state_header[5] = i32::from(loader.should_shuffle); // shuffle state of the dataloader
    // int main state, start at 10 to leave some padding
    state_header[10] = step; // step of the optimization
    // model rng state, start at 20 to leave some padding
    write_u64_as_i32s(&mut state_header, 20, model.rng_state);
    write_u64_as_i32s(&mut state_header, 22, model.rng_state_last_update);
    // dataloader state, start at 30 to leave some padding
    write_u64_as_i32s(&mut state_header, 30, 0usize as u64); // TODO: loader.current_shard_idx
    write_u64_as_i32s(&mut state_header, 32, 0usize as u64); // TODO: loader.current_sample_idx
    
    // Write header
    state_file.write_all(bytemuck::cast_slice(&state_header)).expect("Failed to write state header");

    // write AdamW m, v, and master_weights here (they are all float)
    let buf_size = 32 * 1024 * 1024 / std::mem::size_of::<f32>(); // IO_BUF_SIZE
    
    if let Some(ref m_memory) = model.m_memory {
        llm_rs::cuda_utils::device_to_file(&mut state_file, m_memory.as_slice(), buf_size, &stream);
    }
    if let Some(ref v_memory) = model.v_memory {
        llm_rs::cuda_utils::device_to_file(&mut state_file, v_memory.as_slice(), buf_size, &stream);
    }
    if model.use_master_weights {
        if let Some(ref master_weights) = model.master_weights {
            llm_rs::cuda_utils::device_to_file(&mut state_file, master_weights.as_slice(), buf_size, &stream);
        }
    }

    // write dataloader state if we are using the Permuted version of it
    if loader.should_shuffle {
        let shuffling_state = loader.save_shuffling_state().unwrap();
        state_file.write_all(bytemuck::cast_slice(&[shuffling_state.total_files])).unwrap();
        state_file.write_all(bytemuck::cast_slice(&shuffling_state.shard_indices)).unwrap();
        state_file.write_all(bytemuck::cast_slice(&[shuffling_state.shard_num_samples])).unwrap();
        state_file.write_all(bytemuck::cast_slice(&shuffling_state.intra_shard_indices)).unwrap();
        let c_mt19937_state = llm_rs::random::to_c_state(&shuffling_state.restored_rng);
        state_file.write_all(bytemuck::cast_slice(&[c_mt19937_state])).unwrap();
    }
}


fn load_state(step: &mut i32, model: &mut GPT2, loader: &mut Dataloader, filename: &str) {
    let mut state_file = std::fs::File::open(filename).expect("Failed to open state file");
    let mut state_header = [0i32; 256];
    state_file.read_exact(bytemuck::cast_slice_mut(&mut state_header)).expect("Failed to read state header");
    
    assert_eq!(state_header[0], 20240527); // magic number
    assert_eq!(state_header[1], 1); // version number
    assert_eq!(state_header[2], 1); // number of processes - TODO: multi_gpu_config.num_processes
    assert_eq!(state_header[3], 0); // rank of this process - TODO: multi_gpu_config.process_rank
    let use_master_weights = state_header[4];  // whether we're using fp32 master weights
    let should_shuffle = state_header[5]; // shuffle state of the dataloader
    *step = state_header[10]; // step of the optimization
    
    model.rng_state = bytemuck::cast([state_header[20], state_header[21]]);
    model.rng_state_last_update = bytemuck::cast([state_header[22], state_header[23]]);
    
    let current_shard_idx = bytemuck::cast::<[i32; 2], usize>([state_header[30], state_header[31]]); // shard index
    let current_sample_idx = bytemuck::cast::<[i32; 2], usize>([state_header[32], state_header[33]]); // position in shard

    // read AdamW m, v, master_weights (they are all float)
    // allocate all the needed memory as necessary
    if use_master_weights == 1 && !model.use_master_weights {
        println!("Warning: Master weights are present in state, but not enabled for current run.");
    } else if use_master_weights == 0 && model.use_master_weights {
        eprintln!("Error: Master weights requested, but not present in state file.");
        std::process::exit(1);
    }

    model.init_state = false;      // we just got the state from file, no need to do first-touch init
    assert!(model.m_memory.is_some());
    assert!(model.v_memory.is_some());
    
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).expect("Failed to create stream");
    let buf_size = 32 * 1024 * 1024 / std::mem::size_of::<f32>(); // IO_BUF_SIZE
    
    file_to_device(model.m_memory.as_mut().unwrap(), &mut state_file, buf_size, &stream);
    file_to_device(model.v_memory.as_mut().unwrap(), &mut state_file,  buf_size, &stream);
    
    if model.use_master_weights {
        assert!(model.master_weights.is_some());
        file_to_device(model.master_weights.as_mut().unwrap(), &mut state_file,buf_size, &stream);
        // restore weights from the master weights using the RNG state before last weight update
        model.rng_state = model.rng_state_last_update;
        model.update(0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0, true, &stream);
        model.rng_state = bytemuck::cast([state_header[20], state_header[21]]); // use final RNG state from checkpoint after this
    }

    // revive the DataLoader object and its state
    if should_shuffle == 1 {
        let total_files: usize = 0;
        state_file.read_exact(bytemuck::cast_slice_mut(&mut [total_files])).unwrap();
        let mut shard_indices = vec![0i32; total_files];
        state_file.read_exact(bytemuck::cast_slice_mut(&mut shard_indices)).unwrap();
        let shard_num_samples: usize = 0;
        state_file.read_exact(bytemuck::cast_slice_mut(&mut [shard_num_samples])).unwrap();
        let mut intra_shard_indices = vec![0i32; shard_num_samples];
        state_file.read_exact(bytemuck::cast_slice_mut(&mut intra_shard_indices)).unwrap();
        let c_mt19937_state = llm_rs::random::CMt19937State::empty();
        state_file.read_exact(bytemuck::cast_slice_mut(&mut [c_mt19937_state])).unwrap();
        // Restore the random generator state from the C struct
        let restored_rng = llm_rs::random::from_c_state(&c_mt19937_state);
        
        let shuffling_state = llm_rs::dataloader::ShufflingState {
            total_files,
            shard_indices,
            shard_num_samples,
            intra_shard_indices,
            restored_rng,
        };
        loader.resume_shuffling(shuffling_state);
    }
    loader.resume(current_shard_idx, current_sample_idx);

    // all done
}


// Write checkpoint function that mirrors the C version
fn write_checkpoint(output_log_dir: &str, step: i32, model: &GPT2, train_loader: &Dataloader, multi_gpu_config: &MultiGpuConfig, stream: &Stream) {
    // a checkpoint contains: model weights, optimizer/dataloader state, and a DONE file
    println!("Writing checkpoint at step {}", step);
    let rank = multi_gpu_config.process_rank;
    
    // only rank 0 writes the model file because it is the same across all ranks
    if rank == 0 {
        let model_filename = format!("{}/model_{:08}.bin", output_log_dir, step);
        model.write_checkpoint(&model_filename);
    }
    
    // all ranks write their state file
    let state_filename = format!("{}/state_{:08}_{:05}.bin", output_log_dir, step, rank);
    save_state(&state_filename, step, model, train_loader, &stream);
    
    // DONE file is a signal that this checkpoint as a whole is complete
    // multi_gpu_barrier(multi_gpu_config);
    if rank == 0 {
        let done_filename = format!("{}/DONE_{:08}", output_log_dir, step);
        let _done_file = File::create(&done_filename).expect("Failed to create DONE file");
    }
}

// Delete checkpoint function
fn delete_checkpoint(output_log_dir: &str, step: i32, multi_gpu_config: &MultiGpuConfig) {
    // mirrors write_checkpoint function, cleans up checkpoint from disk
    println!("Deleting checkpoint at step {}", step);

    let rank = multi_gpu_config.process_rank;
    
    // Delete model file (only rank 0)
    if rank == 0 {
        let model_filename = format!("{}/model_{:08}.bin", output_log_dir, step);
        let _ = std::fs::remove_file(&model_filename);
    }
    
    // Delete state file (all ranks)
    let state_filename = format!("{}/state_{:08}_{:05}.bin", output_log_dir, step, rank);
    let _ = std::fs::remove_file(&state_filename);
    
    // Delete DONE file (only rank 0)
    if rank == 0 {
        let done_filename = format!("{}/DONE_{:08}", output_log_dir, step);
        let _ = std::fs::remove_file(&done_filename);
    }
}


#[derive(Parser, Debug)]
#[command(name = "train_gpt2_cuda")]
#[command(about = "Train GPT-2 model with CUDA")]
struct Args {
    #[arg(short = 'i', long, default_value = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin")]
    train_data_pattern: String,

    #[arg(short = 'j', long, default_value = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin")]
    val_data_pattern: String,

    #[arg(short = 'e', long, default_value = "gpt2_124M_bf16.bin")]
    load_filename: String,

    #[arg(short = 'k', long, default_value = "cosine")]
    lr_scheduler_type: String,

    #[arg(short = 'o', long)]
    output_log_dir: Option<String>,

    #[arg(short = 'n', long, default_value = "0")]
    checkpoint_every: i32,  // write checkpoints every how many steps?

    #[arg(long = "nk", default_value = "0")]
    checkpoints_keep: i32,  // how long checkpoint history do we keep? (in units of checkpoints)

    #[arg(long = "nm", default_value = "0")]
    major_checkpoint_every: i32,  // Major checkpoint every N steps

    #[arg(short = 'y', long, default_value = "false")]
    resume: bool,  // resume the optimization, if one is found inside output_log_dir?

    #[arg(short = 'b', long, default_value = "4")]
    batch_size: usize, // batch size

    #[arg(short = 't', long, default_value = "1024")]
    seq_len: usize, // sequence length max

    #[arg(short = 'd', long, default_value = "-1")]
    total_batch_size: i32,  // will be calculated down below later, if not provided

    #[arg(short = 'l', long, default_value = "3e-4")]
    learning_rate: f32,

    #[arg(long = "lg", default_value = "-1")]
    log_gpu_every: i32,

    #[arg(short = 'u', long, default_value = "0")]
    warmup_iterations: i32,

    #[arg(short = 'q', long, default_value = "1.0")]
    final_learning_rate_frac: f32, // final fraction of learning rate, at end of training

    #[arg(short = 'c', long, default_value = "0.0")]
    weight_decay: f32,

    #[arg(long = "sl", default_value = "0.0")]
    skip_update_lossz: f32, // skip update if loss goes above this in zscore

    #[arg(long = "sg", default_value = "0.0")]
    skip_update_gradz: f32, // skip update if grad_norm goes above this in zscore

    #[arg(short = 'v', long, default_value = "20")]
    val_loss_every: i32, // every how many steps do we eval validation loss?

    #[arg(short = 'm', long, default_value = "20")]
    val_max_steps: i32, // how many batches max do we eval for validation loss?

    #[arg(short = 's', long, default_value = "20")]
    sample_every: i32, // every how many steps to do inference?

    #[arg(short = 'g', long, default_value = "64")]
    gen_t: i32, // number of steps of inference we will do

    #[arg(short = 'a', long, default_value = "0")]
    overfit_single_batch: i32, // useful for debugging, 1 = only load a single data batch once

    #[arg(short = 'x', long, default_value = "-1")]
    max_steps: i32,    

    #[arg(short = 'f', long, default_value = "true")]
    override_enable_tf32: bool,

    #[arg(short = 'w', long, default_value = "true")]
    use_master_weights: bool,

    #[arg(long = "ge", default_value = "-1")]
    gelu_fusion: i32, // 0 = none, 1 = forward, 2 = forward+backward (-1 => per-GPU default)

    #[arg(short = 'r', long, default_value = "1")]
    recompute: i32, // recompute during backward setting, 0 = none, 1 = recompute gelu

    #[arg(short = 'z', long, default_value = "0")]
    zero_stage: i32, // Zero Optimization Stage for Multi-GPU training

    #[arg(short = 'h', long, default_value = "0")]
    hellaswag_eval: i32,  // Hellaswag evaluation (0 or 1)

    // multi-node settings
    #[arg(long = "pn", default_value = "1")]
    num_processes: usize,  // this should be set by the slurm environment

    #[arg(long = "pr", default_value = "0")]
    process_rank: i32,  // this should be set by the slurm environment

    #[arg(long = "pg", default_value = "8")]
    gpus_per_node: i32,  // this should be set by the slurm environment

    #[arg(long = "pi", default_value = "mpi")]
    nccl_init_method: String,   // "tcp" or "fs" or "mpi"

    #[arg(long = "ps")]
    server_ip: Option<String>,  // used if init_method set to "tcp" -> set to your server ip address

    #[arg(long = "pf")]
    fs_path: Option<String>,  // used if init_method set to "fs" -> set to a shared filesystem path

}

#[allow(non_snake_case)]
pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read in the (optional) command line arguments
    let args = Args::parse();

    let B = args.batch_size;
    let T = args.seq_len;

    let _context = cust::quick_init()?;
    // multi_gpu_config = multi_gpu_config_init(num_processes, process_rank, gpus_per_node, server_ip, fs_path, nccl_init_method);
    let stream = common_start(args.override_enable_tf32).expect("CUDA init failed");

    let num_processes = args.num_processes;
    
    // Create a simple multi_gpu_config for checkpoint functionality
    let multi_gpu_config = MultiGpuConfig::new(args.process_rank, num_processes, args.zero_stage);

    // should do a bit more error checking here
    assert!(args.warmup_iterations >= 0);
    if let Some(ref output_log_dir) = args.output_log_dir {
        assert!(output_log_dir.len() < 400); // careful bunch of hardcoded snprintf around this
    }
    let tokens_per_fwdbwd = B * T * num_processes; // one micro-batch processes this many tokens
    // calculate sensible default for total batch size as assuming no gradient accumulation
    let mut total_batch_size = args.total_batch_size;
    if total_batch_size == -1 { total_batch_size = tokens_per_fwdbwd as i32; }
    // in the future, we might want to set gelu fusion to 2 for SM90+ and 0 for other GPUs
    let mut gelu_fusion = args.gelu_fusion;
    if gelu_fusion == -1 { gelu_fusion = 0; } // (deviceProp.major >= 9) ? 2 : 0; } // in gpt2_init_common for test_gpt2cu...
    // calculate the number of gradient accumulation steps from the desired total batch size
    assert!(total_batch_size % (tokens_per_fwdbwd as i32) == 0);
    let grad_accum_steps = total_batch_size / (tokens_per_fwdbwd as i32);
    // if we're only overfitting a single batch for debugging, let's overfit the first batch
    // from val instead of train split, because val is smaller and faster. (train_gpt2.py does the same)
    let mut train_data_pattern = args.train_data_pattern.clone();
    if args.overfit_single_batch == 1 { train_data_pattern = args.val_data_pattern.clone(); }
    println!("+-----------------------+----------------------------------------------------+");
    println!("| Parameter             | Value                                              |");
    println!("+-----------------------+----------------------------------------------------+");
    println!("| train data pattern    | {:<50} |", train_data_pattern);
    println!("| val data pattern      | {:<50} |", args.val_data_pattern);
    println!("| output log dir        | {:<50} |", args.output_log_dir.as_deref().unwrap_or("NULL"));
    println!("| checkpoint_every      | {:<50} |", args.checkpoint_every);
    println!("| resume                | {:<50} |", args.resume);
    println!("| micro batch size B    | {:<50} |", B);
    println!("| sequence length T     | {:<50} |", T);
    println!("| total batch size      | {:<50} |", total_batch_size);
    println!("| LR scheduler          | {:<50} |", args.lr_scheduler_type);
    println!("| learning rate (LR)    | {:<50e} |", args.learning_rate);
    println!("| warmup iterations     | {:<50} |", args.warmup_iterations);
    println!("| final LR fraction     | {:<50e} |", args.final_learning_rate_frac);
    println!("| weight decay          | {:<50e} |", args.weight_decay);
    println!("| skip update lossz     | {:<50} |", args.skip_update_lossz);
    println!("| skip update gradz     | {:<50} |", args.skip_update_gradz);
    println!("| max_steps             | {:<50} |", args.max_steps);
    println!("| val_loss_every        | {:<50} |", args.val_loss_every);
    println!("| val_max_steps         | {:<50} |", args.val_max_steps);
    println!("| sample_every          | {:<50} |", args.sample_every);
    println!("| genT                  | {:<50} |", args.gen_t);
    println!("| overfit_single_batch  | {:<50} |", args.overfit_single_batch);
    println!("| use_master_weights    | {:<50} |", if args.use_master_weights { "enabled" } else { "disabled" });
    println!("| gelu_fusion           | {:<50} |", gelu_fusion);
    println!("| recompute             | {:<50} |", args.recompute);
    println!("+-----------------------+----------------------------------------------------+");
    // println!("| device                | {:<50} |", deviceProp.name);
    // println!("| peak TFlops           | {:<50.1} |", get_flops_promised(deviceProp.name, PRECISION_MODE));
    println!("| precision             | {:<50} |", PRECISION_MODE.as_str());
    println!("+-----------------------+----------------------------------------------------+");

    // figure out if we are going to be resuming the optimization
    let mut resuming = false;
    // find the DONE file with the highest step count
    let resume_max_step = find_max_step(args.output_log_dir.as_deref());
    let mut filename_buffer = String::new();
    if args.resume {
        assert!(args.output_log_dir.is_some());
        if resume_max_step != -1 {
            resuming = true; // -y 1 is set, and we found a checkpoint we can resume from
            filename_buffer = format!("{}/model_{:08}.bin", args.output_log_dir.as_ref().unwrap(), resume_max_step);
        }
    }

    // build the GPT-2 model
    let mut model = if resuming {
        // if `-y 1` was set, then we are resuming from the latest checkpoint
        // if we are using master weights, we'll init them later inside load_state()
        let weight_init = !args.use_master_weights;
        GPT2::from_checkpoint(&filename_buffer, weight_init, &stream)
    } else if args.load_filename.ends_with(".bin") {
        // otherwise, if this is a .bin file, we assume it's a model, let's init from it
        GPT2::from_checkpoint(&args.load_filename, true, &stream)
    } else {
        // if it's not .bin, it could be a "special descriptor". This descriptor is used to
        // construct GPT-2 / GPT-3 models in a convenient format. See the function for docs.
        GPT2::from_descriptor(&args.load_filename)
    };
    model.use_master_weights = args.use_master_weights;
    model.gelu_fusion = gelu_fusion;
    model.recompute = args.recompute;

    println!("| weight init method    | {:<50} |", if resuming { "intermediate checkpoint" } else { &args.load_filename });
    println!("| max_sequence_length T | {:<50} |", model.config.max_seq_len);
    println!("| vocab_size V          | {:<50} |", model.config.vocab_size);
    println!("| padded_vocab_size Vp  | {:<50} |", model.config.padded_vocab_size);
    println!("| num_layers L          | {:<50} |", model.config.num_layers);
    println!("| num_heads NH          | {:<50} |", model.config.num_heads);
    println!("| channels C            | {:<50} |", model.config.channels);
    println!("| num_parameters        | {:<50} |", model.num_parameters);
    println!("+-----------------------+----------------------------------------------------+");

    // build DataLoaders for both train and val
    let permute_train_loader = if args.overfit_single_batch == 1 { 0 } else { 1 };
    let mut train_loader = Dataloader::new(&train_data_pattern, B as usize, T as usize, permute_train_loader != 0);
    let mut val_loader = Dataloader::new(&args.val_data_pattern, B as usize, T as usize, false);

    // figure out the number of training steps we will run for
    let mut train_num_batches = args.max_steps; // passed in from command line
    if train_num_batches == -1 {
        // sensible default is to train for exactly one epoch
        let ntok = train_loader.num_tokens;
        // the number of (outer loop) steps each process should take for us to reach one epoch
        train_num_batches = ntok as i32/ total_batch_size;
    }
    // figure out the number of validation steps to run for
    let mut val_num_batches = args.val_max_steps; // passed in from command line
    if val_num_batches == -1 {
        // sensible default is to evaluate the full validation split
        let ntok = val_loader.num_tokens;
        // note that unlike the training loop, there is no gradient accumulation inner loop here
        val_num_batches = (ntok / tokens_per_fwdbwd) as i32;
    }
    println!("| train_num_batches     | {:<50} |", train_num_batches);
    println!("| val_num_batches       | {:<50} |", val_num_batches);
    println!("+-----------------------+----------------------------------------------------+");

    // build an EvalLoader for HellaSwag
    let hellaswag_path = "dev/data/hellaswag/hellaswag_val.bin";
    let hellaswag_available = std::path::Path::new(hellaswag_path).exists();
    let run_hellaswag = args.hellaswag_eval != 0 && hellaswag_available;

    let mut eval_loader = if run_hellaswag {
        Some(EvalLoader::init(hellaswag_path, B, T, 0, 1))
    } else {  None } ;
    
    println!("| run hellaswag         | {:<50} |", if run_hellaswag { "yes" } else { "no" });
    println!("+-----------------------+----------------------------------------------------+");

    // pretty print in a table the multi-gpu configuration as well
    // set_zero_configs(&multi_gpu_config, zero_stage, model.num_parameters);
    println!("| num_processes         | {:<50} |", args.num_processes);
    println!("| zero_stage            | {:<50} |", args.zero_stage);
    println!("+-----------------------+----------------------------------------------------+");

    // prints outside of pretty table to here and below
    if !hellaswag_available {
        println!("HellaSwag eval not found at {}, skipping its evaluation", hellaswag_path);
        println!("You can run `python dev/data/hellaswag.py` to export and use it with `-h 1`.");
    }
    // more prints related to allocations from gpt2_build_from_checkpoint down here to not mess up our table above
    println!("num_parameters: {}", model.num_parameters);
    // few more prints for gradient accumulation math up above
    println!("batch_size B={} * seq_len T={} * num_processes={} and total_batch_size={}", B, T, args.num_processes, args.total_batch_size);
    println!("=> setting grad_accum_steps={}", grad_accum_steps);

    // set up logging
    // if (multi_gpu_config.process_rank == 0) { create_dir_if_not_exists(output_log_dir); }
    //let mut logger = Logger::default();
    // logger_init(&logger, output_log_dir, multi_gpu_config.process_rank, resume);
    if args.process_rank == 0 && args.output_log_dir.is_some() { std::fs::create_dir_all(args.output_log_dir.as_deref().unwrap())?; }
    let logger = Logger::init(args.output_log_dir.as_deref(), args.process_rank, resuming).expect("Failed to initialize logger");

    // set up the Tokenizer
    let tokenizer = Tokenizer::init("gpt2_tokenizer.bin");

    // set up learning rate scheduler
    let lr_scheduler = LearningRateScheduler::new(
        &args.lr_scheduler_type,
        args.learning_rate,
        args.warmup_iterations,
        train_num_batches,
        args.final_learning_rate_frac,
    );

    // some memory for generating samples from the model
    let mut gen_tokens = vec![0i32; (B * T) as usize];
    let mut cpu_logits_raw = vec![zero_floatx(); model.config.vocab_size];
    let mut cpu_logits = vec![0.0f32; model.config.vocab_size];

    // if we found a checkpoint to resume from, load the optimization state
    let mut step = 0;
    model.allocate_state(B, T);
    if resuming {
        let state_filename = format!("{}/state_{:08}_{:05}.bin", args.output_log_dir.as_ref().unwrap(), resume_max_step, args.process_rank);
        load_state(&mut step, &mut model, &mut train_loader, &state_filename);
    }

    // init an OutlierDetector the training loss
    let mut loss_outlier_detector = OutlierDetector::default();
    let mut grad_norm_outlier_detector = OutlierDetector::default();

    // do some checks here before we kick off training
    // cross-check the desired sequence length T with the model's max sequence length
    if T < model.config.max_seq_len {
        println!("!!!!!!!!");
        println!("WARNING:");
        println!("- The training sequence length is: T={} (set with -t)", T);
        println!("- The model's max sequence length is: max_seq_len={}", model.config.max_seq_len);
        println!("You are attempting to train with a sequence length shorter than the model's max.");
        println!("This will lead to unused parameters in the wpe position embedding weights.");
        println!("If you know what you're doing you can ignore this warning.");
        println!("If you're like ???, you are most likely misconfiguring your training run.");
        println!("---> HINT: If you're training GPT-2 use -t 1024. If GPT-3, use -t 2048.");
        println!("!!!!!!!!");
    }
    // in any case, this must be true or we'd index beyond the model's wpe (position embedding table)
    assert!(T <= model.config.max_seq_len);

    // train
    let start_event = Event::new(EventFlags::DEFAULT).unwrap();
    let stop_event = Event::new(EventFlags::DEFAULT).unwrap();
    // cudaCheck(cudaProfilerStart());
    let mut total_sum_iteration_time_s = 0.0;
    let mut ema_tokens_per_second = 0.0f32;
    for step in 0..=train_num_batches {
        // NvtxRange step_range("Train step", step);

        let last_step = step == train_num_batches;

        // once in a while estimate the validation loss (all processes collaborate)
        if step % args.val_loss_every == 0 || last_step {
            // NvtxRange validation_range("validation");
            let mut val_loss = 0.0f32;
            val_loader.reset();
            for _ in 0..val_num_batches {
                val_loader.next_batch();
                val_loss += model.validate(val_loader.inputs(), val_loader.targets(), B, T, &stream);
            }
            val_loss /= val_num_batches as f32;
            // val_loss = multi_gpu_cpu_float_sum(val_loss, &multi_gpu_config) / multi_gpu_config.num_processes;
            println!("val loss {}", val_loss);
            logger.log_val(step, val_loss)?;
        }

        // once in a while estimate HellaSwag accuracy (all processes collaborate)
        if run_hellaswag && ((step > 0 && step % args.val_loss_every == 0) || last_step) {
            println!("HellaSwag");
            let eval_loader = eval_loader.as_mut().ok_or("eval_loader is None")?;
            // NvtxRange evaluation_range("evaluation");
            let mut eval_acc_norm = 0.0f32;
            eval_loader.reset();
            for i in 0..eval_loader.num_batches {
                if i % 10 == 0 { print!("evaluating HellaSwag: {}/{}\r", i, eval_loader.num_batches); }
                eval_loader.next_batch();
                model.validate(eval_loader.inputs.as_slice(), eval_loader.targets.as_slice(), B, T, &stream);
                let correct = eval_loader.stat_losses(model.cpu_losses.as_ref().unwrap());
                eval_acc_norm += correct as f32;
            }
            // careful because not all ranks may have the exact same allocation of number of examples
            // eval_acc_norm = multi_gpu_cpu_float_sum(eval_acc_norm, &multi_gpu_config);
            println!("HellaSwag: {}/{} = {}", eval_acc_norm as i32, eval_loader.num_examples, eval_acc_norm / eval_loader.num_examples as f32);
            logger.log_eval(step, eval_acc_norm / eval_loader.num_examples as f32)?;
        }

        // once in a while do model inference to print generated text (only rank 0)
        // if (multi_gpu_config.process_rank == 0 && sample_every > 0 && (step > 0 && (step % sample_every) == 0 || last_step)) {
        if args.sample_every > 0 && (step > 0 && (step % args.sample_every) == 0 || last_step) {
            // NvtxRange generation_range("generation");
            let mut sample_rng_state = 1337u64;
            // fill up gen_tokens with the <|endoftext|> token, which kicks off the generation
            let eot_token = tokenizer.eot_token;
            for i in 0..(B * T) as usize {
                gen_tokens[i] = eot_token;
            }
            // now sample from the model autoregressively
            println!("generating:");
            println!("---");
            for t in 1..args.gen_t {
                // NvtxRange generation_range("Generation step", t);

                // we try not to be too wasteful for inference by not calculating all of B,T
                // Using a smaller B is always bit-for-bit identical, but T is more tricky
                // for non-CUDNN, we need to make sure the attention buffer is memset to 0
                // for cuDNN, it might suddenly decide to use a slightly different algorithm...
                // on cuDNN 9.2.1 with cuDNN FrontEnd 1.5.2, T >= 256 seems bit-for-bit identical
                // (but even if it wasn't fully identical that's probably not the end of the world)
                // note this is still somewhat wasteful because we don't have a KV cache!
                let t_max_256 = min(T,256);
                let t_gen = (t as usize).div_ceil(t_max_256) * t_max_256;
                model.forward(gen_tokens.as_slice(), 1, t_gen, &stream);
                // get the V-dimensional vector probs[0, t-1, :]
                let logits_offset = (t as usize - 1) * model.config.padded_vocab_size;
                let logits = model.acts.as_ref().unwrap().output.index(logits_offset .. logits_offset + model.config.vocab_size);
                // move probs back to CPU and sample
                logits.copy_to(&mut cpu_logits_raw[..model.config.vocab_size]).unwrap();
                // convert to FP32 into cpu_logits (this does nothing useful if floatX == float)
                for i in 0..model.config.vocab_size {
                    cpu_logits[i] = cpu_logits_raw[i].to_f32();
                }
                // sample the next token
                let coin = sampler::random_f32(&mut sample_rng_state);
                let next_token = sampler::sample_softmax(&cpu_logits, coin) as i32;
                gen_tokens[t as usize] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                print!("{}", tokenizer.decode(next_token));
                std::io::stdout().flush().unwrap(); // TODO necessary ?
            }
            println!();
            println!("---");
        }

        // once in a while checkpoint the optimization state (all ranks)
        if (args.checkpoint_every > 0 && args.output_log_dir.is_some() && !resuming) && ((step > 0 && step % args.checkpoint_every == 0) || last_step) {
            println!("write checkpoint");
            // writes model .bin file, state .bin files, and DONE file for step
            let output_log_dir = args.output_log_dir.as_ref().unwrap();
            write_checkpoint(output_log_dir, step, &model, &train_loader, &multi_gpu_config, &stream);
            // we only keep checkpoints_keep checkpoints on disk to save space
            // so now that we wrote a new checkpoint, delete one old one (unless it is a "major" checkpoint)
            // we only do this is checkpoint keeping is turned on (checkpoints_keep > 0)
            let step_delete = step - args.checkpoints_keep * args.checkpoint_every;
            if args.checkpoints_keep > 0 && step_delete > 0 && (args.major_checkpoint_every == 0 || step_delete % args.major_checkpoint_every != 0) {
                delete_checkpoint(output_log_dir, step_delete, &multi_gpu_config);
            }
        }
        resuming = false;

        // bit confusing: we want to make sure to eval and sample on 0th iteration
        // but also after the very last iteration. so we loop for step <= train_num_batches
        // instead of just < train_num_batches (one extra due to <=), only to do
        // the validation/sampling one last time, and then we break right here as we're done.
        if last_step { break; }

        // --------------- TRAINING SECTION BEGIN -----------------
        if args.overfit_single_batch == 1 {
            // if we are trying to overfit a single batch, we reset the loader here
            train_loader.reset();
        }
        // do one training step, doing forward/backward/update on total_batch_size tokens
        start_event.record(&stream).unwrap();
        // gradient and loss accumulation loop over micro-batches
        for micro_step in 0..grad_accum_steps {
            // fetch the next data batch
            train_loader.next_batch();
            // forward pass. note that we pass in grad_accum_steps, which scales down the loss
            model.forward(train_loader.inputs(), B, T, &stream);
            // backward pass. all model params accumulate gradients with += inside this inner loop
            model.backward_and_reduce(train_loader.inputs(), train_loader.targets(), grad_accum_steps as usize, micro_step as usize, &stream);
        }
        let zloss = loss_outlier_detector.update(model.mean_loss as f64) as f32; // loss z-score
        // fetch the next learning rate
        let step_learning_rate = lr_scheduler.get_learning_rate(step);
        // calculate the gradient norm and how much we wish to scale the gradient
        let grad_norm = model.calculate_grad_norm(&multi_gpu_config, &stream);
        let zgrad = grad_norm_outlier_detector.update(grad_norm as f64) as f32; // grad z-score
        // update the model parameters
        if zloss.is_finite() && args.skip_update_lossz != 0.0f32 && zloss > args.skip_update_lossz {
            println!("skipping update due to loss z-score of {}", zloss);
        } else if zgrad.is_finite() && args.skip_update_gradz != 0.0f32 && zgrad > args.skip_update_gradz {
            println!("skipping update due to grad z-score of {}", zgrad);
        } else {
            // clip the gradient norm to a maximum value
            let grad_clip = 1.0f32;
            let grad_scale = if grad_norm > grad_clip { grad_clip / grad_norm } else { 1.0f32 };
            model.update(step_learning_rate, 0.9f32, 0.95f32, 1e-8f32, args.weight_decay, grad_scale, step+1,  false, &stream);
        }
        stop_event.record(&stream).unwrap();
        stop_event.synchronize().unwrap(); // wait for the end event to finish to get correct timings
        // --------------- TRAINING SECTION END -------------------
        // everything that follows now is just diagnostics, prints, logging, etc.

        // todo - move or double-buffer all of this timing logic to avoid idling the GPU at this point!
        let time_elapsed_ms = stop_event.elapsed_time_f32(&start_event).unwrap();
        let tokens_processed = multi_gpu_config.num_processes * B * T * grad_accum_steps as usize;
        let tokens_per_second = tokens_processed as f32 / time_elapsed_ms * 1000.0f32;
        let mut bias_corrected_ema_tokens_per_second = tokens_per_second; // by default set to non-ema version
        if step > 0 { // consider the first batch to be a warmup (e.g. cuBLAS/cuDNN initialisation)
            total_sum_iteration_time_s += time_elapsed_ms / 1000.0f32;
            // smooth out the tok/s with an exponential moving average, and bias correct just like in AdamW
            ema_tokens_per_second = 0.95f32 * ema_tokens_per_second + 0.05f32 * tokens_per_second;
            bias_corrected_ema_tokens_per_second = ema_tokens_per_second / (1.0f32 - 0.95f32.powf(step as f32));
        }
        let mfu = model.estimate_mfu(B * T * grad_accum_steps as usize, time_elapsed_ms / 1000.0f32);
        println!("step {:4}/{} | loss {:7.6} ({:+2}z)| norm {:6.4} ({:+2}z)| lr {:.2e} | {:.2} ms | {:.1}% bf16 MFU | {:.0} tok/s",
                step + 1, train_num_batches, model.mean_loss, zloss, grad_norm, zgrad, step_learning_rate,
                time_elapsed_ms, 100.0*mfu, bias_corrected_ema_tokens_per_second);
        if args.log_gpu_every > 0 && (step + 1) % args.log_gpu_every == 0 {
            if let Ok(gpu_info) = llm_rs::gpu_monitor::get_gpu_utilization_info() {
                println!("                  compute {:2.1}% | memory: {:2.1}% | fan: {:2}% | {:4} MHz / {:4} MHz | {:3} W / {:3} W | {}°C / {}°C | {}",
                        gpu_info.gpu_utilization, gpu_info.mem_utilization, gpu_info.fan, gpu_info.clock, gpu_info.max_clock, gpu_info.power / 1000, gpu_info.power_limit / 1000,
                        gpu_info.temperature, gpu_info.temp_slowdown, gpu_info.throttle_reason);
            } else {
                println!("                  Failed to get GPU utilization info");
            }
        }
        logger.log_train(step, model.mean_loss, step_learning_rate, grad_norm)?;

        // disable the profiler after 3 steps of optimization
        if step == 3 { 
            // cudaProfilerStop(); 
        }
    }
    // add a total average, for optimizations that are only mild improvements (excluding 1st batch as warmup)
    println!("total average iteration time: {} ms", total_sum_iteration_time_s / (train_num_batches-1) as f32 * 1000.0);

    Ok(())
}
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    train_gpt2_cuda::main()
}
