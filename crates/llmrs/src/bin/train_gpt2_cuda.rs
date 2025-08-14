use llm_rs::common::{f32_to_floatx, zero_floatx, Bf16, FloatX, PrecisionMode, F16, PRECISION_MODE};
use clap::Parser;
use std::io::Write;
use std::fs::File;
use std::io::BufReader;
use llm_rs::utils::{find_max_step, read_le_u32_array};
use llm_rs::{dataloader::Dataloader, dataloader::EvalLoader, tokenizer::Tokenizer, scheduler::LearningRateScheduler};
use cust::{prelude::*, stream};


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

struct ParameterTensors<'a> {
    wte: &'a mut [FloatX], // (V, C)
    wpe: &'a mut [FloatX], // (maxT, C)
    ln1w: &'a mut [FloatX], // (L, C)
    ln1b: &'a mut [FloatX], // (L, C)
    qkvw: &'a mut [FloatX], // (L, 3*C, C)
    qkvb: &'a mut [FloatX], // (L, 3*C)
    attprojw: &'a mut [FloatX], // (L, C, C)
    attprojb: &'a mut [FloatX], // (L, C)
    ln2w: &'a mut [FloatX], // (L, C)
    ln2b: &'a mut [FloatX], // (L, C)
    fcw: &'a mut [FloatX], // (L, 4*C, C)
    fcb: &'a mut [FloatX], // (L, 4*C)
    fcprojw: &'a mut [FloatX], // (L, C, 4*C)
    fcprojb: &'a mut [FloatX], // (L, C)
    lnfw: &'a mut [FloatX], // (C)
    lnfb: &'a mut [FloatX], // (C)
}

// the parameters of the model
pub const NUM_ACTIVATION_TENSORS: usize = 21;

struct ActivationTensors<'a> {
    encoded: &'a mut [FloatX], // (B, T, C)
    ln1: &'a mut [FloatX],     // (L, B, T, C)
    ln1_mean: &'a mut [FloatX],// (L, B, T)
    ln1_rstd: &'a mut [FloatX],// (L, B, T)
    
    att: &'a mut [FloatX],  // (L, B, NH, T, T)
    residual2: &'a mut [FloatX],  // (L, B, T, C)
    ln2: &'a mut [FloatX],  // (L, B, T, C)
    ln2_mean: &'a mut [f32], // (L, B, T)
    ln2_rstd: &'a mut [f32], // (L, B, T)
    fch: &'a mut [FloatX], // (L, B, T, 4*C)
    fch_gelu: &'a mut [FloatX], // (L, B, T, 4*C)
    residual3: &'a mut [FloatX],  // (L, B, T, C)
    lnf: &'a mut [FloatX],  // (B, T, C);   if LN recomputation is enabled (-r 2 and above), will be used for _all_ layernorms
    lnf_mean: &'a mut [f32],  // (B, T)
    lnf_rstd: &'a mut [f32],  // (B, T)
    losses: &'a mut [f32],  // (B, T), will be accumulated in micro-steps
    // adding these two compared to the CPU .c code, needed for attention kernel as buffers
    qkvr: &'a mut [FloatX], // (L, B, T, 3*C)
    // in inference mode, this buffer will store the logits
    // in training mode, this buffer will contain the *gradients* of the logits.
    // during the processing of transformer blocks, we will also use this as a
    // general scratchpad buffer. Allocation is made large enough to hold (B, T, 3C),
    // (B, NH, T, T), and (B, T, V) shaped tensors.
    output: &'a mut [FloatX],

    // some additional scratch buffers
    scratch_bt4c: &'a mut [FloatX],   // (B, T, 4*C)
    scratch_btc: &'a mut [FloatX],    // (B, T, C)
}



struct GPT2 {
    config: GPT2Config,

    // the weights of the model, and their sizes
    //params are obtained from params_memory with scoped mutable borrow
    param_sizes: [usize; NUM_PARAMETER_TENSORS],
    params_memory: DeviceBuffer<FloatX>,
    num_parameters: usize,

    // gradients of the weights
    //grads are obtained from grads_memory with scoped mutable borrow
    grads_memory: Option<Vec<f32>>,
    // buffers for the AdamW optimizer
    m_memory: Option<Vec<f32>>,
    v_memory: Option<Vec<f32>>,
    // float* master_weights;     // is NULL unless fp32 weights is enabled.
    // the activations of the model, and their sizes
    //ActivationTensors acts;
    act_sizes: [usize; NUM_ACTIVATION_TENSORS],
    acts_memory: Option<Vec<f32>>,
    
    batch_size: usize, // the batch size (B) of current forward pass
    seq_len: usize, // the sequence length (T) of current forward pass
    inputs: Option<Vec<u32>>, // the input tokens for the current forward pass
    targets: Option<Vec<u32>>, // the target tokens for the current forward pass
    mean_loss: f32,  // after the last backward micro-batch, will be populated with mean loss across all GPUs and micro-steps

    accumulated_mean_loss: Option<*mut f32>,  // GPU buffer used to accumulate loss across micro-steps
    cpu_losses: Option<*mut f32>, // CPU buffer to copy the losses to, allocated with cudaMallocHost
    rng_state: u64, // the RNG state for seeding stochastic rounding etc.
    rng_state_last_update: u64, // RNG before last gpt2_update() to re-round identically from master weights
    use_master_weights: bool, // keep master weights copy in float for optim update?
    init_state: bool,   // set to true if master weights need to be initialized
    gelu_fusion: i32, // fuse gelu via cuBLASLt (0=none, 1=forward, 2=forward+backward)
    recompute: i32, // recompute gelu | layernorm forward during model backward? 0|1|2
    // CPU scratch buffers for encoder backward
    workload_indices: Option<*mut i32>, // encoder_backward, B*T*num_c_groups (int)
    bucket_info: Option<*mut i32>,     // encoder_backward, B*T*num_c_groups (int4) - size for worst case
}

impl GPT2 {
    /*
    /// Create a new GPT2 model with default configuration
    pub fn new() -> Self {
        GPT2 {
            config: GPT2Config {
                max_seq_len: 0,
                vocab_size: 0,
                padded_vocab_size: 0,
                num_layers: 0,
                num_heads: 0,
                channels: 0,
            },
            param_sizes: [0; NUM_PARAMETER_TENSORS],
            params_memory: Vec::new(),
            num_parameters: 0,
            grads_memory: None,
            m_memory: None,
            v_memory: None,
            act_sizes: [0; NUM_ACTIVATION_TENSORS],
            acts_memory: None,
            batch_size: 0,
            seq_len: 0,
            inputs: None,
            targets: None,
            mean_loss: -1.0, // -1.0 designates no loss, set at end of forward()
            accumulated_mean_loss: None,
            cpu_losses: None,
            rng_state: 13371337, // used in stochastic rounding
            rng_state_last_update: 0,
            use_master_weights: true, // safe default: do keep master weights in fp32
            init_state: true,
            gelu_fusion: 0, // default: off for now
            recompute: 1, // good default: recompute gelu but not layernorm
            workload_indices: None,
            bucket_info: None,
        }
    } */

    fn allocate_weights(config: &GPT2Config) -> ([usize; NUM_PARAMETER_TENSORS], usize, DeviceBuffer<FloatX>) {
        let param_sizes = GPT2::fill_in_parameter_sizes(&config);
        let num_parameters = param_sizes.iter().sum();

        let params_memory: DeviceBuffer<FloatX> = DeviceBuffer::zeroed(num_parameters).unwrap();
        
        (param_sizes, num_parameters, params_memory)
    }

    #[allow(non_snake_case)]
    fn fill_in_parameter_sizes(config: &GPT2Config) -> [usize; NUM_PARAMETER_TENSORS] {
        let Vp = config.padded_vocab_size;
        let C = config.channels;
        let maxT = config.max_seq_len;
        let L = config.num_layers;

        [ Vp * C,   // wte
        maxT * C,   // wpe
        L * C,      // ln1w
        L * C,      // ln1b
        L * (3 * C) * C,    // qkvw
        L * (3 * C),        // qkvb
        L * C * C,  // attprojw
        L * C,      // attprojb
        L * C,      // ln2w
        L * C,      // ln2b
        L * (4 * C) * C,    // fcw
        L * (4 * C),        // fcb
        L * C * (4 * C),    // fcprojw
        L * C,              // fcprojb
        C, // lnfw
        C, // lnfb
        ]
    }

    fn alloc_params_buffer(param_sizes: [usize; NUM_PARAMETER_TENSORS]) -> Vec<f32> {
        let total_size: usize = param_sizes.iter().sum();
        vec![0f32; total_size]
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
        let (param_sizes, num_params, mut params_memory) = GPT2::allocate_weights(&config);

        // read in the parameters if weight_init is true
        if weight_init {
            llm_rs::utils::file_to_device(&mut params_memory, &mut model_file_reader, 1024, &stream);
        }

        // only return from this function once we are certain the params are ready on the GPU
        stream.synchronize().unwrap();

        Self {
            config: config,
            param_sizes: param_sizes,
            params_memory: params_memory,
            grads_memory: None,
            m_memory: None,
            v_memory: None,
            num_parameters: num_params,
            act_sizes: [0; NUM_ACTIVATION_TENSORS],
            acts_memory: None,
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

        let (param_sizes, num_params, mut params_memory) = GPT2::allocate_weights(&config);

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
                    let ones = f32_to_floatx(&vec![1.0f32; param_sizes[i]]);
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
                    let floatx_buffer = f32_to_floatx(&fp32_buffer);
                    params_memory_cpu[offset + layer_offset ..offset + layer_offset  + n].copy_from_slice(&floatx_buffer);
                }
                offset += param_sizes[i];
            }
        }
        
        // copy them to GPU
        params_memory.copy_from(&params_memory_cpu).unwrap();
        
        Self {
            config: config,
            param_sizes: param_sizes,
            params_memory: params_memory,
            grads_memory: None,
            m_memory: None,
            v_memory: None,
            num_parameters: num_params,
            act_sizes: [0; NUM_ACTIVATION_TENSORS],
            acts_memory: None,
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
}


fn common_start(_override_enable_tf32: bool) -> cust::error::CudaResult<Stream> {
    let _context = cust::quick_init()?;
    let device_idx = 0;
    let _device = Device::get_device(device_idx)?;
    println!("Device #{}, Name: {}", device_idx, _device.name()?);

    // TODO setup cublas

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    Ok(stream)
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
    batch_size: i32, // batch size

    #[arg(short = 't', long, default_value = "1024")]
    seq_len: i32, // sequence length max

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
    num_processes: i32,  // this should be set by the slurm environment

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
fn main() {
    // read in the (optional) command line arguments
    let args = Args::parse();

    let B = args.batch_size;
    let T = args.seq_len;


    // multi_gpu_config = multi_gpu_config_init(num_processes, process_rank, gpus_per_node, server_ip, fs_path, nccl_init_method);
    let stream = common_start(args.override_enable_tf32).expect("CUDA init failed");

    let num_processes = args.num_processes;

    // should do a bit more error checking here
    assert!(args.warmup_iterations >= 0);
    if let Some(ref output_log_dir) = args.output_log_dir {
        assert!(output_log_dir.len() < 400); // careful bunch of hardcoded snprintf around this
    }
    let tokens_per_fwdbwd = B * T * num_processes; // one micro-batch processes this many tokens
    // calculate sensible default for total batch size as assuming no gradient accumulation
    let mut total_batch_size = args.total_batch_size;
    if total_batch_size == -1 { total_batch_size = tokens_per_fwdbwd; }
    // in the future, we might want to set gelu fusion to 2 for SM90+ and 0 for other GPUs
    let mut gelu_fusion = args.gelu_fusion;
    if gelu_fusion == -1 { gelu_fusion = 0; } // (deviceProp.major >= 9) ? 2 : 0; } // in gpt2_init_common for test_gpt2cu...
    // calculate the number of gradient accumulation steps from the desired total batch size
    assert!(total_batch_size % tokens_per_fwdbwd == 0);
    let grad_accum_steps = total_batch_size / tokens_per_fwdbwd;
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
    println!("| gelu_fusion           | {:<50} |", args.gelu_fusion);
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
    // dataloader_init(&train_loader, train_data_pattern, B, T, multi_gpu_config.process_rank, multi_gpu_config.num_processes, permute_train_loader);
    // dataloader_init(&val_loader, val_data_pattern, B, T, multi_gpu_config.process_rank, multi_gpu_config.num_processes, 0);

    // figure out the number of training steps we will run for
    let mut train_num_batches = args.max_steps; // passed in from command line
    if train_num_batches == -1 {
        // sensible default is to train for exactly one epoch
        let ntok = train_loader.num_tokens;
        // the number of (outer loop) steps each process should take for us to reach one epoch
        train_num_batches = (ntok / args.total_batch_size as usize) as i32;
    }
    // figure out the number of validation steps to run for
    let mut val_num_batches = args.val_max_steps; // passed in from command line
    if val_num_batches == -1 {
        // sensible default is to evaluate the full validation split
        let ntok = val_loader.num_tokens;
        // note that unlike the training loop, there is no gradient accumulation inner loop here
        val_num_batches = (ntok / tokens_per_fwdbwd as usize) as i32;
    }
    println!("| train_num_batches     | {:<50} |", train_num_batches);
    println!("| val_num_batches       | {:<50} |", val_num_batches);
    println!("+-----------------------+----------------------------------------------------+");

    // build an EvalLoader for HellaSwag
    let mut eval_loader = EvalLoader::default();
    let hellaswag_path = "dev/data/hellaswag/hellaswag_val.bin";
    let hellaswag_available = std::path::Path::new(hellaswag_path).exists();
    let run_hellaswag = args.hellaswag_eval != 0 && hellaswag_available;
    if run_hellaswag {
        // evalloader_init(&eval_loader, hellaswag_path, B, T, multi_gpu_config.process_rank, multi_gpu_config.num_processes);
    }
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
    println!("num_parameters: {} => bytes: {}", model.num_parameters, model.num_parameters * std::mem::size_of::<f32>());
    println!("allocated {} MiB for model parameters", (model.num_parameters * std::mem::size_of::<f32>() / (1024 * 1024)) as i32);
    // few more prints for gradient accumulation math up above
    println!("batch_size B={} * seq_len T={} * num_processes={} and total_batch_size={}", B, T, args.num_processes, args.total_batch_size);
    println!("=> setting grad_accum_steps={}", grad_accum_steps);

    // set up logging
    // if (multi_gpu_config.process_rank == 0) { create_dir_if_not_exists(output_log_dir); }
    //let mut logger = Logger::default();
    // logger_init(&logger, output_log_dir, multi_gpu_config.process_rank, resume);

    // set up the Tokenizer
    let mut tokenizer = Tokenizer::init("gpt2_tokenizer.bin");

    // set up learning rate scheduler
    let mut lr_scheduler = LearningRateScheduler::new(
        &args.lr_scheduler_type,
        args.learning_rate,
        args.warmup_iterations,
        train_num_batches,
        args.final_learning_rate_frac,
    );

    // some memory for generating samples from the model
    let mut gen_tokens = vec![0; (B * T) as usize];
    let mut cpu_logits_raw = vec![0.0f32; model.config.vocab_size];
    let mut cpu_logits = vec![0.0f32; model.config.vocab_size];

    // if we found a checkpoint to resume from, load the optimization state
    let mut step = 0;
    // gpt2_allocate_state(&model, B, T);
    if resuming {
        // snprintf(filename_buffer, sizeof(filename_buffer), "%s/state_%08d_%05d.bin", output_log_dir, resume_max_step, multi_gpu_config.process_rank);
        // load_state(&step, &model, &train_loader, filename_buffer);
    }

    // init an OutlierDetector the training loss
    // TODO: Implement OutlierDetector in Rust
    // let mut loss_outlier_detector = OutlierDetector::default();
    // let mut grad_norm_outlier_detector = OutlierDetector::default();
    // init_detector(&loss_outlier_detector);
    // init_detector(&grad_norm_outlier_detector);

    // do some checks here before we kick off training
    // cross-check the desired sequence length T with the model's max sequence length
    if T < model.config.max_seq_len as i32 {
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
    assert!(T <= model.config.max_seq_len as i32);

    // train
    // cudaEvent_t start, end;
    // cudaCheck(cudaEventCreate(&start));
    // cudaCheck(cudaEventCreate(&end));
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
            // dataloader_reset(&val_loader);
            for i in 0..val_num_batches {
                // dataloader_next_batch(&val_loader);
                // val_loss += gpt2_validate(&model, val_loader.inputs, val_loader.targets, B, T);
            }
            val_loss /= val_num_batches as f32;
            // val_loss = multi_gpu_cpu_float_sum(val_loss, &multi_gpu_config) / multi_gpu_config.num_processes;
            println!("val loss {}", val_loss);
            // logger_log_val(&logger, step, val_loss);
        }

        // once in a while estimate HellaSwag accuracy (all processes collaborate)
        if run_hellaswag && ((step > 0 && step % args.val_loss_every == 0) || last_step) {
            // NvtxRange evaluation_range("evaluation");
            let mut eval_acc_norm = 0.0f32;
            // evalloader_reset(&eval_loader);
            for i in 0..eval_loader.num_batches {
                if i % 10 == 0 { print!("evaluating HellaSwag: {}/{}\r", i, eval_loader.num_batches); }
                // evalloader_next_batch(&eval_loader);
                // gpt2_validate(&model, eval_loader.inputs, eval_loader.targets, B, T);
                // let correct = evalloader_stat_losses(&eval_loader, model.cpu_losses);
                // eval_acc_norm += correct as f32;
            }
            // careful because not all ranks may have the exact same allocation of number of examples
            // eval_acc_norm = multi_gpu_cpu_float_sum(eval_acc_norm, &multi_gpu_config);
            println!("HellaSwag: {}/{} = {}", eval_acc_norm as i32, eval_loader.num_examples, eval_acc_norm / eval_loader.num_examples as f32);
            // logger_log_eval(&logger, step, eval_acc_norm / eval_loader.num_examples as f32);
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
                // gpt2_forward(&model, gen_tokens, 1, CEIL_DIV(t, min(T,256)) * min(T,256));
                // get the V-dimensional vector probs[0, t-1, :]
                // let logits = model.acts.output + (t - 1) * model.config.padded_vocab_size;
                // move probs back to CPU and sample (note we only move the first vocab_size logits, ignoring the padding)
                // cudaCheck(cudaMemcpy(cpu_logits_raw, logits, model.config.vocab_size * sizeof(floatX), cudaMemcpyDeviceToHost));
                // convert to FP32 into cpu_logits (this does nothing useful if floatX == float)
                for i in 0..model.config.vocab_size {
                    cpu_logits[i] = cpu_logits_raw[i] as f32;
                }
                // sample the next token
                // TODO: Implement random_f32 and sample_softmax functions
                // let coin = random_f32(&mut sample_rng_state);
                // let next_token = sample_softmax(&cpu_logits, model.config.vocab_size, coin);
                let next_token = 0; // placeholder
                gen_tokens[t as usize] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                // TODO: Implement tokenizer_decode function
                // if tokenizer.init_ok {
                //     let token_str = tokenizer_decode(&tokenizer, next_token);
                //     print!("{}", token_str);
                // } else {
                //     // fall back to printing the token id
                //     print!("{} ", next_token);
                // }
                print!("{} ", next_token); // fallback for now
                std::io::stdout().flush().unwrap();
            }
            println!();
            println!("---");
        }

        // once in a while checkpoint the optimization state (all ranks)
        if (args.checkpoint_every > 0 && args.output_log_dir.is_some() && !resuming) && ((step > 0 && step % args.checkpoint_every == 0) || last_step) {
            // writes model .bin file, state .bin files, and DONE file for step
            // write_checkpoint(output_log_dir, step, &model, &train_loader, &multi_gpu_config);
            // we only keep checkpoints_keep checkpoints on disk to save space
            // so now that we wrote a new checkpoint, delete one old one (unless it is a "major" checkpoint)
            // we only do this is checkpoint keeping is turned on (checkpoints_keep > 0)
            let step_delete = step - args.checkpoints_keep * args.checkpoint_every;
            if args.checkpoints_keep > 0 && step_delete > 0 && (args.major_checkpoint_every == 0 || step_delete % args.major_checkpoint_every != 0) {
                // delete_checkpoint(output_log_dir, step_delete, &multi_gpu_config);
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
            // dataloader_reset(&train_loader);
        }
        // do one training step, doing forward/backward/update on total_batch_size tokens
        // cudaCheck(cudaEventRecord(start));
        // gradient and loss accumulation loop over micro-batches
        for micro_step in 0..grad_accum_steps {
            // fetch the next data batch
            // dataloader_next_batch(&train_loader);
            // forward pass. note that we pass in grad_accum_steps, which scales down the loss
            // gpt2_forward(&model, train_loader.inputs, B, T);
            // backward pass. all model params accumulate gradients with += inside this inner loop
            // gpt2_backward_and_reduce(&model, train_loader.inputs, train_loader.targets, grad_accum_steps, micro_step);
        }
        // let zloss = update_detector(&loss_outlier_detector, model.mean_loss as f64) as f32; // loss z-score
        // fetch the next learning rate
        let step_learning_rate = lr_scheduler.get_learning_rate(step);
        // calculate the gradient norm and how much we wish to scale the gradient
        // let grad_norm = gpt2_calculate_grad_norm(&model, &multi_gpu_config);
        // let zgrad = update_detector(&grad_norm_outlier_detector, grad_norm as f64) as f32; // grad z-score
        // update the model parameters
        // if (isfinite(zloss) && skip_update_lossz != 0.0f && zloss > skip_update_lossz) {
        //     println!("skipping update due to loss z-score of {}", zloss);
        // } else if (isfinite(zgrad) && skip_update_gradz != 0.0f && zgrad > skip_update_gradz) {
        //     println!("skipping update due to grad z-score of {}", zgrad);
        // } else {
        //     // clip the gradient norm to a maximum value
        //     let grad_clip = 1.0f32;
        //     let grad_scale = if grad_norm > grad_clip { grad_clip / grad_norm } else { 1.0f32 };
        //     gpt2_update(&model, step_learning_rate, 0.9f32, 0.95f32, 1e-8f32, weight_decay, grad_scale, step+1, &multi_gpu_config);
        // }
        // cudaCheck(cudaEventRecord(end));
        // cudaCheck(cudaEventSynchronize(end)); // wait for the end event to finish to get correct timings
        // --------------- TRAINING SECTION END -------------------
        // everything that follows now is just diagnostics, prints, logging, etc.

        // todo - move or double-buffer all of this timing logic to avoid idling the GPU at this point!
        // float time_elapsed_ms;
        // cudaCheck(cudaEventElapsedTime(&time_elapsed_ms, start, end));
        // size_t tokens_processed = (size_t)multi_gpu_config.num_processes * B * T * grad_accum_steps;
        // float tokens_per_second = tokens_processed / time_elapsed_ms * 1000.0f;
        // float bias_corrected_ema_tokens_per_second = tokens_per_second; // by default set to non-ema version
        // if (step > 0) { // consider the first batch to be a warmup (e.g. cuBLAS/cuDNN initialisation)
        //     total_sum_iteration_time_s += time_elapsed_ms / 1000.0f;
        //     // smooth out the tok/s with an exponential moving average, and bias correct just like in AdamW
        //     ema_tokens_per_second = 0.95f * ema_tokens_per_second + 0.05f * tokens_per_second;
        //     bias_corrected_ema_tokens_per_second = ema_tokens_per_second / (1.0f - powf(0.95f, step));
        // }
        // float mfu = gpt2_estimate_mfu(&model, B * T * grad_accum_steps, time_elapsed_ms / 1000.0f);
        // println!("step {:4}/{} | loss {:7.6} ({:+2}z)| norm {:6.4} ({:+2}z)| lr {:.2e} | {:.2} ms | {:.1}% bf16 MFU | {:.0} tok/s",
        //         step + 1, train_num_batches, model.mean_loss, zloss, grad_norm, zgrad, step_learning_rate,
        //         time_elapsed_ms, 100.0*mfu, bias_corrected_ema_tokens_per_second);
        // if(log_gpu_every > 0 && (step + 1) % log_gpu_every == 0) {
        //     GPUUtilInfo gpu_info = get_gpu_utilization_info();
        //     println!("                  compute {:2.1}% | memory: {:2.1}% | fan: {:2}% | {:4} MHz / {:4} MHz | {:3} W / {:3} W | {}°C / {}°C | {}",
        //             gpu_info.gpu_utilization, gpu_info.mem_utilization, gpu_info.fan, gpu_info.clock, gpu_info.max_clock, gpu_info.power / 1000, gpu_info.power_limit / 1000,
        //             gpu_info.temperature, gpu_info.temp_slowdown, gpu_info.throttle_reason);
        // }
        // logger_log_train(&logger, step, model.mean_loss, step_learning_rate, grad_norm);

        // disable the profiler after 3 steps of optimization
        if step == 3 { 
            // cudaProfilerStop(); 
        }
    }
    // add a total average, for optimizations that are only mild improvements (excluding 1st batch as warmup)
    println!("total average iteration time: {} ms", total_sum_iteration_time_s / (train_num_batches-1) as f64 * 1000.0);

    // free and destroy everything
    // cudaCheck(cudaEventDestroy(end));
    // cudaCheck(cudaEventDestroy(start));
    if run_hellaswag { 
        // evalloader_free(&eval_loader); 
    }
    // dataloader_free(&train_loader);
    // dataloader_free(&val_loader);
    // tokenizer_free(&tokenizer);
    // multi_gpu_config_free(&multi_gpu_config);
    // gpt2_free(&model);
    // common_free(model);
}
