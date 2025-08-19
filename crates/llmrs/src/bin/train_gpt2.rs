use std::{fs::File, io::BufReader, time::Instant};
use llm_rs::{dataloader::Dataloader, tokenizer::Tokenizer, utils::{read_fill_le_f32_array, read_le_u32_array}};


#[derive(Debug)]
struct GPT2Config {
    max_seq_len: usize, // max sequence length, e.g. 1024
    vocab_size: usize, // vocab size, e.g. 50257
    padded_vocab_size: usize, // padded to e.g. %128==0, 50304
    num_layers: usize, // number of layers, e.g. 12
    num_heads: usize,  // number of heads in attention, e.g. 12
    channels: usize, // number of channels, e.g. 768

}

// the parameters of the model
pub const NUM_PARAMETER_TENSORS: usize = 16;

struct ParameterTensors<'a> {
    wte: &'a mut [f32], // (V, C)
    wpe: &'a mut [f32], // (maxT, C)
    ln1w: &'a mut [f32], // (L, C)
    ln1b: &'a mut [f32], // (L, C)
    qkvw: &'a mut [f32], // (L, 3*C, C)
    qkvb: &'a mut [f32], // (L, 3*C)
    attprojw: &'a mut [f32], // (L, C, C)
    attprojb: &'a mut [f32], // (L, C)
    ln2w: &'a mut [f32], // (L, C)
    ln2b: &'a mut [f32], // (L, C)
    fcw: &'a mut [f32], // (L, 4*C, C)
    fcb: &'a mut [f32], // (L, 4*C)
    fcprojw: &'a mut [f32], // (L, C, 4*C)
    fcprojb: &'a mut [f32], // (L, C)
    lnfw: &'a mut [f32], // (C)
    lnfb: &'a mut [f32], // (C)
}

// the parameters of the model
pub const NUM_ACTIVATION_TENSORS: usize = 23;

struct ActivationTensors<'a> {
    encoded: &'a mut [f32], // (B, T, C)
    ln1: &'a mut [f32],     // (L, B, T, C)
    ln1_mean: &'a mut [f32],// (L, B, T)
    ln1_rstd: &'a mut [f32],// (L, B, T)
    qkv: &'a mut [f32],     // (L, B, T, 3*C)
    atty: &'a mut [f32],    // (L, B, T, C)
    preatt: &'a mut [f32],  // (L, B, NH, T, T)
    att: &'a mut [f32],
    attproj: &'a mut [f32],
    residual2: &'a mut [f32],
    ln2: &'a mut [f32],
    ln2_mean: &'a mut [f32],
    ln2_rstd: &'a mut [f32],
    fch: &'a mut [f32],
    fch_gelu: &'a mut [f32],
    fcproj: &'a mut [f32],
    residual3: &'a mut [f32],
    lnf: &'a mut [f32],
    lnf_mean: &'a mut [f32],
    lnf_rstd: &'a mut [f32],
    logits: &'a mut [f32],
    probs: &'a mut [f32],
    losses: &'a mut [f32],
    
}


struct GPT2 {
    config: GPT2Config,
    //params: ParameterTensors<'a>, // the weights (parameters) of the model, and their sizes

    param_sizes: [usize; NUM_PARAMETER_TENSORS],
    params_memory: Vec<f32>,
    num_parameters: usize,
    // gradients of the weights
    //grads: ParameterTensors,
    grads_memory: Option<Vec<f32>>,
    // buffers for the AdamW optimizer
    m_memory: Option<Vec<f32>>,
    v_memory: Option<Vec<f32>>,
    // the activations of the model, and their sizes
    //ActivationTensors acts;
    act_sizes: [usize; NUM_ACTIVATION_TENSORS],
    acts_memory: Option<Vec<f32>>,
    //num_activations: usize,
    // gradients of the activations
    //ActivationTensors grads_acts;
    grads_acts_memory: Option<Vec<f32>>,
    // other run state configuration
    batch_size: usize, // the batch size (B) of current forward pass
    seq_len: usize, // the sequence length (T) of current forward pass
    inputs: Option<Vec<i32>>, // the input tokens for the current forward pass
    targets: Option<Vec<i32>>, // the target tokens for the current forward pass
    mean_loss: f32, // after a forward pass with targets, will be populated with the mean loss
}


impl GPT2 {

    #[allow(non_snake_case)]
    pub fn build_from_checkpoint(checkpoint_path: &str) -> Self {

        let model_file = File::open(checkpoint_path)
        .unwrap_or_else(|_| panic!("Error: cannot open file {:?}", checkpoint_path));
        let mut model_file_reader = BufReader::new(model_file);
        let model_header: [u32; 256] = read_le_u32_array::<_, 256>(&mut model_file_reader);
        if model_header[0] != 20240326 { panic!("Bad magic model file"); }
        if model_header[1] != 3 {
            print!("Bad version in model file");
            print!("---> HINT: try to re-run `python train_gpt2.py");
            panic!();
        }

        // read in hyperparameters
        let maxT = model_header[2] as usize;
        let V = model_header[3] as usize;
        let L = model_header[4] as usize;
        let NH = model_header[5] as usize;
        let C = model_header[6] as usize;
        let Vp = model_header[7] as usize;

        let config = GPT2Config {
            max_seq_len: maxT,
            vocab_size: V,
            num_layers: L,
            num_heads: NH,
            channels: C,
            padded_vocab_size: Vp
        };
        println!("[GPT-2]");
        println!("max_seq_len: {}", maxT);
        println!("vocab_size: {}", V);
        println!("padded_vocab_size: {}", Vp);
        println!("num_layers: {}", L);
        println!("num_heads: {}", NH);
        println!("channels: {}", C);

        // allocate space for all the parameters and read them in
        let param_sizes = GPT2::fill_in_parameter_sizes(&config);

        // count the number of parameters
        let num_parameters = param_sizes.iter().sum();
        println!("num_parameters: {}", num_parameters);


        // read in all the parameters from file
        let mut params_memory = GPT2::alloc_params_buffer(param_sizes);
        read_fill_le_f32_array(&mut model_file_reader, &mut params_memory);
        //let params = GPT2::point_parameters(param_sizes, &mut params_memory);
        
        Self {
            config: config,
            param_sizes: param_sizes,
            params_memory: params_memory,
            grads_memory: None,
            m_memory: None,
            v_memory: None,
            num_parameters: num_parameters,
            act_sizes: [0; NUM_ACTIVATION_TENSORS],
            acts_memory: None,
            grads_acts_memory: None,
            batch_size: 0,
            seq_len: 0,
            inputs: None,
            targets: None,
            mean_loss: -1.0  // -1.0f will designate no loss
        }


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

    fn get_parameters(params_memory: &mut [f32], param_sizes: [usize; NUM_PARAMETER_TENSORS]) -> ParameterTensors<'_> {
        let param_sizes = param_sizes;
        // Rust impl: use split_at_mut to avoid the usage of unsafe

        let (wte, rest) = params_memory.split_at_mut(param_sizes[0]);
        let (wpe, rest) = rest.split_at_mut(param_sizes[1]);
        let (ln1w, rest) = rest.split_at_mut(param_sizes[2]);
        let (ln1b, rest) = rest.split_at_mut(param_sizes[3]);
        let (qkvw, rest) = rest.split_at_mut(param_sizes[4]);
        let (qkvb, rest) = rest.split_at_mut(param_sizes[5]);
        let (attprojw, rest) = rest.split_at_mut(param_sizes[6]);
        let (attprojb, rest) = rest.split_at_mut(param_sizes[7]);
        let (ln2w, rest) = rest.split_at_mut(param_sizes[8]);
        let (ln2b, rest) = rest.split_at_mut(param_sizes[9]);
        let (fcw, rest) = rest.split_at_mut(param_sizes[10]);
        let (fcb, rest) = rest.split_at_mut(param_sizes[11]);
        let (fcprojw, rest) = rest.split_at_mut(param_sizes[12]);
        let (fcprojb, rest) = rest.split_at_mut(param_sizes[13]);
        let (lnfw, rest) = rest.split_at_mut(param_sizes[14]);
        let (lnfb, rest) = rest.split_at_mut(param_sizes[15]);
        debug_assert_eq!(rest.len(), 0, "Parameter buffer not fully consumed: {} elements left", rest.len());

        ParameterTensors {
            wte: wte,
            wpe: wpe,
            ln1w: ln1w,
            ln1b: ln1b,
            qkvw: qkvw,
            qkvb: qkvb,
            attprojw: attprojw,
            attprojb: attprojb,
            ln2w: ln2w,
            ln2b: ln2b,
            fcw: fcw,
            fcb: fcb,
            fcprojw: fcprojw,
            fcprojb: fcprojb,
            lnfw: lnfw,
            lnfb: lnfb
        }

    }

    #[allow(non_snake_case)]
    fn fill_in_activation_sizes(config: &GPT2Config, B: usize, T: usize) -> [usize; NUM_ACTIVATION_TENSORS] {
        let C = config.channels;
        let NH = config.num_heads;
        let L = config.num_layers;
        let Vp = config.padded_vocab_size;

        [
            B * T * C, // encoded
            L * B * T * C, // ln1
            L * B * T, // ln1_mean
            L * B * T, // ln1_rstd
            L * B * T * 3 * C, // qkv
            L * B * T * C, // atty
            L * B * NH * T * T, // preatt
            L * B * NH * T * T, // att
            L * B * T * C, // attproj
            L * B * T * C, // residual2
            L * B * T * C, // ln2
            L * B * T, // ln2_mean
            L * B * T, // ln2_rstd
            L * B * T * 4 * C, // fch
            L * B * T * 4 * C, // fch_gelu
            L * B * T * C, // fcproj
            L * B * T * C, // residual3
            B * T * C, // lnf
            B * T, // lnf_mean
            B * T, // lnf_rstd
            B * T * Vp, // logits
            B * T * Vp, // probs
            B * T, // losses
        ]
    }

    fn alloc_activations_buffer(activations_sizes: [usize; NUM_ACTIVATION_TENSORS]) -> Vec<f32> {
        let total_size: usize = activations_sizes.iter().sum();
        vec![0f32; total_size]
    }

    fn get_activations(acts_memory: &mut [f32], act_sizes: [usize; NUM_ACTIVATION_TENSORS]) -> ActivationTensors<'_> {
        // split_at_mut is one way to avoid the usage of unsafe

        let (encoded, rest) = acts_memory.split_at_mut(act_sizes[0]);
        let (ln1, rest) = rest.split_at_mut(act_sizes[1]);
        let (ln1_mean, rest) = rest.split_at_mut(act_sizes[2]);
        let (ln1_rstd, rest) = rest.split_at_mut(act_sizes[3]);
        let (qkv, rest) = rest.split_at_mut(act_sizes[4]);
        let (atty, rest) = rest.split_at_mut(act_sizes[5]);
        let (preatt, rest) = rest.split_at_mut(act_sizes[6]);
        let (att, rest) = rest.split_at_mut(act_sizes[7]);
        let (attproj, rest) = rest.split_at_mut(act_sizes[8]);
        let (residual2, rest) = rest.split_at_mut(act_sizes[9]);
        let (ln2, rest) = rest.split_at_mut(act_sizes[10]);
        let (ln2_mean, rest) = rest.split_at_mut(act_sizes[11]);
        let (ln2_rstd, rest) = rest.split_at_mut(act_sizes[12]);
        let (fch, rest) = rest.split_at_mut(act_sizes[13]);
        let (fch_gelu, rest) = rest.split_at_mut(act_sizes[14]);
        let (fcproj, rest) = rest.split_at_mut(act_sizes[15]);
        let (residual3, rest) = rest.split_at_mut(act_sizes[16]);
        let (lnf, rest) = rest.split_at_mut(act_sizes[17]);
        let (lnf_mean, rest) = rest.split_at_mut(act_sizes[18]);
        let (lnf_rstd, rest) = rest.split_at_mut(act_sizes[19]);
        let (logits, rest) = rest.split_at_mut(act_sizes[20]);
        let (probs, rest) = rest.split_at_mut(act_sizes[21]);
        let (losses, rest) = rest.split_at_mut(act_sizes[22]);
        debug_assert_eq!(rest.len(), 0, "Activation buffer not fully consumed: {} elements left", rest.len());


        ActivationTensors { 
            encoded: encoded,
            ln1: ln1,
            ln1_mean: ln1_mean,
            ln1_rstd: ln1_rstd,
            qkv: qkv,
            atty: atty,
            preatt: preatt,
            att: att,
            attproj: attproj,
            residual2: residual2,
            ln2: ln2,
            ln2_mean: ln2_mean,
            ln2_rstd: ln2_rstd,
            fch: fch,
            fch_gelu: fch_gelu, 
            fcproj: fcproj, 
            residual3: residual3, 
            lnf: lnf, 
            lnf_mean: lnf_mean, 
            lnf_rstd: lnf_rstd, 
            logits: logits, 
            probs: probs, 
            losses: losses }
    }

    #[allow(non_snake_case)]
    pub fn forward(&mut self, inputs: &[i32], opt_targets: Option<&[i32]>, B: usize, T: usize) {

        // convenience parameters
        let V = self.config.vocab_size;
        let Vp = self.config.padded_vocab_size;
        let L = self.config.num_layers;
        let NH = self.config.num_heads;
        let C = self.config.channels;

        // validate inputs, all indices must be in the range [0, V)
        inputs.iter().for_each(|&input| assert!((input as usize) < V));
        if let Some(targets) = opt_targets {
            targets.iter().for_each(|&target| assert!((0..V).contains(&(target as usize))));
        }

        // allocate space for all the activations if needed (done here, lazily)
        if self.acts_memory.is_none() {
            // record the current B,T as well
            self.batch_size = B;
            self.seq_len = T;

            self.act_sizes = GPT2::fill_in_activation_sizes(&self.config, B, T);
            println!("num_activations: {}", self.act_sizes.iter().sum::<usize>());
            self.acts_memory = Some(GPT2::alloc_activations_buffer(self.act_sizes));
        }
        

        // cache the inputs/targets
        self.inputs = Some(inputs.to_vec());
        self.targets = opt_targets.map(|t| t.to_vec());

        // forward pass
        let (params_memory, acts_memory) = (&mut self.params_memory, &mut self.acts_memory.as_mut().unwrap());
        let params = GPT2::get_parameters(params_memory, self.param_sizes);
        let mut acts = GPT2::get_activations(acts_memory, self.act_sizes);

        encoder_forward(&mut acts.encoded, inputs, params.wte, params.wpe, B, T, C);
        for l in 0..L {
            let residual = if l == 0 { 
                &mut acts.encoded 
            } else {
                &mut acts.residual3[(l-1) * B * T * C .. l * B * T * C]
            };

            // get the pointers of the weights for this layer
            let l_ln1w = &params.ln1w[l*C .. (l+1)*C];
            let l_ln1b = &params.ln1b[l*C .. (l+1)*C];
            let l_qkvw = &params.qkvw[l*3*C*C .. (l+1)*3*C*C];
            let l_qkvb = &params.qkvb[l*3*C .. (l+1)*3*C];
            let l_attprojw = &params.attprojw[l*C*C .. (l+1)*C*C];
            let l_attprojb = &params.attprojb[l*C .. (l+1)*C];
            let l_ln2w = &params.ln2w[l*C .. (l+1)*C];
            let l_ln2b = &params.ln2b[l*C .. (l+1)*C];
            let l_fcw = &params.fcw[l*4*C*C .. (l+1)*4*C*C];
            let l_fcb = &params.fcb[l*4*C .. (l+1)*4*C];
            let l_fcprojw = &params.fcprojw[l*C*4*C .. (l+1)*C*4*C];
            let l_fcprojb = &params.fcprojb[l*C .. (l+1)*C];
            
            // get the pointers of the activations for this layer
            let l_ln1 = &mut acts.ln1[l*B*T*C .. (l+1)*B*T*C];
            let l_ln1_mean = &mut acts.ln1_mean[l*B*T .. (l+1)*B*T];
            let l_ln1_rstd = &mut acts.ln1_rstd[l*B*T .. (l+1)*B*T];
            let l_qkv = &mut acts.qkv[l*B*T*3*C .. (l+1)*B*T*3*C];
            let l_atty = &mut acts.atty[l*B*T*C .. (l+1)*B*T*C];
            let l_preatt = &mut acts.preatt[l*B*NH*T*T .. (l+1)*B*NH*T*T];
            let l_att = &mut acts.att[l*B*NH*T*T .. (l+1)*B*NH*T*T];
            let l_attproj = &mut acts.attproj[l*B*T*C .. (l+1)*B*T*C];
            let l_residual2 = &mut acts.residual2[l*B*T*C .. (l+1)*B*T*C];
            let l_ln2 = &mut acts.ln2[l*B*T*C .. (l+1)*B*T*C];
            let l_ln2_mean = &mut acts.ln2_mean[l*B*T .. (l+1)*B*T];
            let l_ln2_rstd = &mut acts.ln2_rstd[l*B*T .. (l+1)*B*T];
            let l_fch = &mut acts.fch[l*B*T*4*C .. (l+1)*B*T*4*C];
            let l_fch_gelu = &mut acts.fch_gelu[l*B*T*4*C .. (l+1)*B*T*4*C];
            let l_fcproj = &mut acts.fcproj[l*B*T*C .. (l+1)*B*T*C];
            

            // now do the forward pass
            layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
            matmul_forward(l_qkv, l_ln1, l_qkvw, Some(l_qkvb), B, T, C, 3*C);
            attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
            matmul_forward(l_attproj, l_atty, l_attprojw, Some(l_attprojb), B, T, C, C);
            residual_forward(l_residual2, residual, l_attproj, B*T*C);
            layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
            matmul_forward(l_fch, l_ln2, l_fcw, Some(l_fcb), B, T, C, 4*C);
            gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
            matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, Some(l_fcprojb), B, T, 4*C, C);
            {
                let l_residual3 = &mut acts.residual3[l * B * T * C .. (l+1)*B*T*C];
                residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
            } 
        }

        let residual = &acts.residual3[(L-1) * B * T * C .. ];  // last residual is in residual3
        layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
        matmul_forward(acts.logits, acts.lnf, params.wte, None, B, T, C, Vp);
        softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

        match opt_targets {
            Some(targets) => {
                crossentropy_forward(acts.losses, acts.probs, targets, B, T, Vp);
                let mut mean_loss = acts.losses.iter().sum::<f32>();
                mean_loss /= (B*T) as f32;
                self.mean_loss = mean_loss;
            },
            None => {
                self.mean_loss = -1.0;
            }
        }

    }

    fn zero_grad(&mut self) {
        if let Some(grads_memory) = self.grads_memory.as_mut() {
            grads_memory.fill(0.0);
        }
        if let Some(grads_acts_memory) = self.grads_acts_memory.as_mut() {
            grads_acts_memory.fill(0.0);
        }
    }

    #[allow(non_snake_case)]
    fn backward(&mut self) {
        if self.mean_loss == -1.0 {
            panic!("Error: must forward with targets before backward");
        }

        // lazily allocate the memory for gradients of the weights and activations, if needed
        if self.grads_memory.is_none() {
            self.grads_memory = Some(GPT2::alloc_params_buffer(self.param_sizes));
            self.grads_acts_memory = Some(GPT2::alloc_activations_buffer(self.act_sizes));
        }

        // convenience shortcuts
        let B = self.batch_size as usize;
        let T = self.seq_len as usize;
        let V = self.config.vocab_size;
        let Vp = self.config.padded_vocab_size;
        let L = self.config.num_layers;
        let NH = self.config.num_heads;
        let C = self.config.channels;

        // backward pass: go in the reverse order of the forward pass, and call backward() functions

        // Rust impl: we must borrow multiple mut references of self at once
        let (params_memory, grads_memory, acts_memory, grads_acts_memory) = 
                (&mut self.params_memory,
                 &mut self.grads_memory.as_mut().unwrap(),
                 &mut self.acts_memory.as_mut().unwrap(),
                 &mut self.grads_acts_memory.as_mut().unwrap()
                );
        let params = GPT2::get_parameters(params_memory, self.param_sizes);
        let grads = GPT2::get_parameters(grads_memory, self.param_sizes);
        let acts = GPT2::get_activations(acts_memory, self.act_sizes);
        let grads_acts = GPT2::get_activations(grads_acts_memory, self.act_sizes);

        // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
        // technically this is a small, inline backward() pass of calculating
        // total, final loss as the mean over all losses over all (B,T) positions in the batch
        let dloss_mean = 1.0 / ((B*T) as f32);
        for i in 0..B*T { grads_acts.losses[i] = dloss_mean;}

        crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, self.targets.as_ref().unwrap(), B, T, V, Vp);
        matmul_backward(grads_acts.lnf, grads.wte, None, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
        let residual = &acts.residual3[(L-1)*B*T*C ..]; // last layer's residual
        let dresidual= &mut grads_acts.residual3[(L-1)*B*T*C ..];  // write to last layer's residual
        layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

        for l in (0..L).rev() {
            // get the pointers of the weights for this layer
            let l_ln1w = &params.ln1w[l*C .. (l+1)*C];
            let l_qkvw = &params.qkvw[l*3*C*C .. (l+1)*3*C*C];
            let l_attprojw = &params.attprojw[l*C*C .. (l+1)*C*C];
            let l_ln2w = &params.ln2w[l*C .. (l+1)*C];
            let l_fcw = &params.fcw[l*4*C*C .. (l+1)*4*C*C];
            let l_fcprojw = &params.fcprojw[l*C*4*C.. (l+1)*C*4*C];
            // get the pointers of the gradients of the weights for this layer
            let dl_ln1w = &mut grads.ln1w[l * C .. (l+1)*C];
            let dl_ln1b = &mut grads.ln1b[l * C .. (l+1)*C];
            let dl_qkvw = &mut grads.qkvw[l * 3*C * C .. (l+1) * 3*C * C];
            let dl_qkvb = &mut grads.qkvb[l * 3*C .. (l+1)*3*C];
            let dl_attprojw = &mut grads.attprojw[l * C * C .. (l+1)*C*C];
            let dl_attprojb = &mut grads.attprojb[l * C .. (l+1)*C];
            let dl_ln2w = &mut grads.ln2w[l * C .. (l+1)*C];
            let dl_ln2b = &mut grads.ln2b[l * C .. (l+1)*C];
            let dl_fcw = &mut grads.fcw[l * 4*C * C .. (l+1)*4*C*C];
            let dl_fcb = &mut grads.fcb[l * 4*C .. (l+1)*4*C];
            let dl_fcprojw = &mut grads.fcprojw[l * C * 4*C .. (l+1)*C*4*C];
            let dl_fcprojb = &mut grads.fcprojb[l * C .. (l+1)*C];
            // get the pointers of the activations for this layer
            let l_ln1 = &acts.ln1[l*B*T*C .. (l+1)*B*T*C];
            let l_ln1_mean = & acts.ln1_mean[l*B*T .. (l+1)*B*T];
            let l_ln1_rstd = & acts.ln1_rstd[l*B*T .. (l+1)*B*T];
            let l_qkv = & acts.qkv[l * B * T * 3*C .. (l+1) * B * T * 3*C];
            let l_atty = & acts.atty[l * B * T * C .. (l+1) * B * T * C];
            let l_att = & acts.att[l * B * NH * T * T .. (l+1) * B * NH * T * T];
            let l_residual2 = & acts.residual2[l * B * T * C .. (l+1) * B * T * C];
            let l_ln2 = & acts.ln2[l * B * T * C .. (l+1) * B * T * C];
            let l_ln2_mean = & acts.ln2_mean[l * B * T .. (l+1) * B * T];
            let l_ln2_rstd = & acts.ln2_rstd[l * B * T .. (l+1) * B * T];
            let l_fch = & acts.fch[l * B * T * 4*C .. (l+1) * B * T * 4*C];
            let l_fch_gelu = & acts.fch_gelu[l * B * T * 4*C .. (l+1) * B * T * 4*C];
            // get the pointers of the gradients of the activations for this layer
            let dl_ln1 = &mut grads_acts.ln1[l * B * T * C .. (l+1) * B * T * C];
            let dl_qkv = &mut grads_acts.qkv[l * B * T * 3*C .. (l+1) * B * T * 3*C];
            let dl_atty = &mut grads_acts.atty[l * B * T * C .. (l+1) * B * T * C];
            let dl_preatt = &mut grads_acts.preatt[l * B * NH * T * T .. (l+1) * B * NH * T * T];
            let dl_att = &mut grads_acts.att[l * B * NH * T * T .. (l+1) * B * NH * T * T];
            let dl_attproj = &mut grads_acts.attproj[l * B * T * C .. (l+1) * B * T * C];
            let dl_residual2 = &mut grads_acts.residual2[l * B * T * C .. (l+1) * B * T * C];
            let dl_ln2 = &mut grads_acts.ln2[l * B * T * C .. (l+1) * B * T * C];
            let dl_fch = &mut grads_acts.fch[l * B * T * 4*C .. (l+1) * B * T * 4*C];
            let dl_fch_gelu = &mut grads_acts.fch_gelu[l * B * T * 4*C .. (l+1) * B * T * 4*C];
            let dl_fcproj = &mut grads_acts.fcproj[l * B * T * C .. (l+1) * B * T * C];
            

            // backprop this layer
            {
                let dl_residual3 = &mut grads_acts.residual3[l * B * T * C .. ];
                residual_backward(dl_residual2, dl_fcproj, dl_residual3, B*T*C);
            }

            // Rust impl: Avoid double mut ref on grads_acts.residual3
            let (residual, dresidual) = if l == 0 { 
                (&acts.encoded[..], &mut grads_acts.encoded[..])
            } else {
                (&acts.residual3[(l-1) * B * T * C ..], &mut grads_acts.residual3[(l-1) * B * T * C ..])
            };
            
            matmul_backward(dl_fch_gelu, dl_fcprojw, Some(dl_fcprojb), dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
            gelu_backward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C);
            matmul_backward(dl_ln2, dl_fcw, Some(dl_fcb), dl_fch, l_ln2, l_fcw, B, T, C, 4*C);
            layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
            residual_backward(dresidual, dl_attproj, dl_residual2, B*T*C);
            matmul_backward(dl_atty, dl_attprojw, Some(dl_attprojb), dl_attproj, l_atty, l_attprojw, B, T, C, C);
            attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);  
            matmul_backward(dl_ln1, dl_qkvw, Some(dl_qkvb), dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C);
            layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C); 
            
        }
        encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, self.inputs.as_ref().unwrap(), B, T, C);
    }

    fn update(&mut self, learning_rate: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32, t: u32) {
        // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

        let grads_memory = self.grads_memory.as_ref().unwrap();

        if self.m_memory.is_none() {
            self.m_memory = Some(vec![0f32; self.num_parameters]);
            self.v_memory = Some(vec![0f32; self.num_parameters]);
        }
        let m_memory = &mut self.m_memory.as_mut().unwrap();
        let v_memory = &mut self.v_memory.as_mut().unwrap();

        for i in 0..self.num_parameters {
            let param = self.params_memory[i];
            let grad = grads_memory[i];

            // update the first moment (momentum)
            let m = beta1 * m_memory[i] + (1.0 - beta1) * grad;
            // update the second moment (RMSprop)
            let v = beta2 * v_memory[i] + (1.0 - beta2) * grad * grad;
            // bias-correct both moments
            let m_hat = m / (1.0 - beta1.powi(t as i32));
            let v_hat = v / (1.0 - beta2.powi(t as i32));

            // update
            m_memory[i] = m;
            v_memory[i] = v;
            self.params_memory[i] -= learning_rate * (m_hat / ((v_hat).sqrt() + eps) + weight_decay * param);
        }
    }

}

#[allow(non_snake_case)]
fn encoder_forward(out: &mut [f32], inp: &[i32], wte: &mut[f32], wpe: &mut[f32], B: usize, T: usize, C: usize) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"

    for b in 0..B {
        for t in 0..T {
            // seek to the output position in out[b,t,:]
            let out_bt = &mut out[b*T*C + t*C .. ];
            // get the index of the token at inp[b, t]
            let ix = inp[b*T + t] as usize;
            // seek to the position in wte corresponding to the token
            let wte_ix = &wte[ix * C .. ];
            // seek to the position in wpe corresponding to the position
            let wpe_t = &wpe[t * C .. ];

            // add the two vectors and store the result in out[b,t,:]
            for i in 0..C {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

#[allow(non_snake_case)]
fn encoder_backward(dwte: &mut [f32], dwpe: &mut [f32], dout: &[f32], inp: &[i32], B: usize, T: usize, C: usize) {
    for b in 0..B {
        for t in 0..T {
            let dout_bt = &dout[b * T * C + t * C .. ];
            let ix = inp[b*T + t] as usize;
            let dwte_ix = &mut dwte[ix * C .. ];
            let dwpe_t = &mut dwpe[t * C .. ];
            for i in 0..C {
                let d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

#[allow(non_snake_case)]
fn layernorm_forward(out: &mut [f32], mean: &mut[f32], rstd: &mut[f32],
                     inp: &[f32], weight: &[f32], bias: &[f32],
                     B: usize, T: usize, C: usize) {

    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted

    let eps = 1e-5f32;
    for b in 0..B {
        for t in 0..T {
            // seek to the input position inp[b,t,:]
            let seek_bt_ = b*T*C + t*C;
            let x    = &inp[seek_bt_ .. seek_bt_ + C];

            // calculate the mean
            let m = x.iter().sum::<f32>() / (C as f32);

            // calculate the variance (without any bias correction)
            let mut v = 0.0f32;
            for i in x {
                let xshift = i - m;
                v += xshift * xshift;
            }
            v = v / (C as f32);

            // calculate the rstd (reciprocal standard deviation)
            let s = 1.0f32 / (v + eps).sqrt();

            // seek to the output position in out[b,t,:]
            for i in 0..C {
                let n = s * (x[i] - m);  // normalize
                let o = n * weight[i] + bias[i]; // scale and shift
                out[seek_bt_ + i] = o; // write
            }

            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

#[allow(non_snake_case)]
fn layernorm_backward(dinp: &mut [f32], dweight: &mut [f32], dbias: &mut [f32],
                      dout: &[f32], inp: &[f32], weight: &[f32], mean: &[f32], rstd: &[f32],
                      B: usize, T: usize, C: usize) {
    for b in 0..B {
        for t in 0..T {
            let btc = b*T*C + t*C;
            let bt = b*T + t;
            let dout_bt = &dout[btc .. ];
            let inp_bt = &inp[btc .. ];
            let dinp_bt = &mut dinp[btc .. ];
            let mean_bt = mean[bt];
            let rstd_bt = rstd[bt];

            // first: two reduce operations
            let mut dnorm_mean = 0.0f32;
            let mut dnorm_norm_mean = 0.0f32;
            for i in 0..C {
                let norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                let dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean /= C as f32;
            dnorm_norm_mean /= C as f32;

            // now iterate again and accumulate all the gradients
            for i in 0..C {
                let norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                let dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                let mut dval = 0.0f32;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

#[allow(non_snake_case)]
fn matmul_forward_naive(out: &mut [f32],
                        inp: &[f32], weight: &[f32], bias: Option<&[f32]>,
                        B: usize, T: usize, C: usize, OC: usize) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.

    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    for b in 0..B {
        for t in 0..T {
            let bt = b*T + t;
            for o in 0..OC {
                let mut val  = match bias {
                    Some(bias_vec) => bias_vec[o],
                    None => 0.0,
                };
                for i in 0..C {
                    val += inp[bt*C + i] * weight[o*C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}

#[allow(non_snake_case)]
fn matmul_forward(out: &mut [f32], 
                  inp: &[f32], weight: &[f32], bias: Option<&[f32]>,
                  B: usize, T: usize, C: usize, OC: usize) {
    const LOOP_UNROLL: usize = 8;
    let BT = B * T;
    if BT % LOOP_UNROLL != 0 {
        // Fallback to naive version
        println!("Warning: fallback to matmul_forward_naive");
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        return;
    }
    
    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    let mut obt = 0;
    while obt < BT {
        for o in 0..OC {
            let mut result = [0.0f32; LOOP_UNROLL];
            // Initialize with bias if present
            if let Some(bias_vec) = bias {
                for ibt in 0..LOOP_UNROLL {
                    result[ibt] = bias_vec[o];
                }
            }
            // Main multiply-accumulate
            for i in 0..C {
                let w = weight[i + o * C];
                for ibt in 0..LOOP_UNROLL {
                    let bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }
            // Write back results
            for ibt in 0..LOOP_UNROLL {
                let bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
        obt += LOOP_UNROLL;
    }
}


#[allow(non_snake_case)]
fn matmul_backward(dinp: &mut [f32], dweight: &mut [f32], mut dbias: Option<&mut [f32]>,
                   dout: &[f32], inp: &[f32], weight: &[f32],
                   B: usize, T: usize, C: usize, OC: usize ) {

    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    for b in 0..B {
        for t in 0..T {
            let dout_bt = &dout[b*T*OC + t*OC .. b*T*OC + t*OC + OC];
            let dinp_bt = &mut dinp[b*T*C + t*C .. b*T*C + t*C + C];
            for o in 0..OC {
                let wrow = &weight[o*C .. o*C + C];
                let d = dout_bt[o];
                for i in 0..C {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }

    // backward into weight/bias, parallelize over output channels OC
    for o in 0..OC {
        for b in 0..B {
            for t in 0..T {
                let dout_bt = &dout[b*T*OC + t*OC .. b*T*OC + t*OC + OC];
                let inp_bt = &inp[b*T*C + t*C .. b*T*C + t*C + C];
                let dwrow = &mut dweight[o*C .. o*C + C];
                let d = dout_bt[o];
                if let Some(dbias) = dbias.as_mut() {
                    dbias[o] += d;
                }
                for i in 0..C {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

#[allow(non_snake_case)]
fn attention_forward(out: &mut [f32], preatt: &mut [f32], att: &mut [f32],
                     inp: &[f32],
                     B: usize, T: usize, C: usize, NH: usize) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)

    let C3 = 3*C;
    let hs = C / NH;
    let scale = 1.0 / (hs as f32).sqrt();

    for b in 0..B {
        for t in 0..T {
            for h in 0..NH {
                let query_bt = &inp[b*T*C3 + t*C3 + h*hs .. ];
                let preatt_bth = &mut preatt[b*NH*T*T + h*T*T + t*T .. ];
                let att_bth = &mut att[b*NH*T*T + h*T*T + t*T .. ];

                // pass 1: calculate query dot key and maxval
                let mut maxval = -10000.0f32; // TODO something better
                for t2 in 0..=t {
                    let key_t2 = &inp[b*T*C3 + t2*C3 + h*hs + C .. ];  // +C because it's key

                    // (query_t) dot (key_t2)
                    let mut val = 0.0f32;
                    for i in 0..hs {
                        val += query_bt[i] * key_t2[i];
                    }
                    val *= scale;
                    if val > maxval {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                let mut expsum = 0.0f32;
                for t2 in 0..=t {
                    let expv = (preatt_bth[t2] - maxval).exp();
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                let expsum_inv = if expsum == 0.0 { 0.0} else { 1.0 /expsum};

                // pass 3: normalize to get the softmax
                for t2 in 0..T {
                    if t2 <= t {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                let out_bth = &mut out[b*T*C + t*C + h*hs .. ];
                for i in 0..hs { out_bth[i] = 0.0; }
                for t2 in 0..=t {
                    let value_t2 = &inp[b*T*C3 + t2*C3 + h*hs + C*2 .. ]; // +C*2 because it's value
                    let att_btht2 = att_bth[t2];
                    for i in 0..hs {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }

}

#[allow(non_snake_case)]
fn attention_backward(dinp: &mut [f32], dpreatt: &mut [f32], datt: &mut [f32],
                      dout: &[f32], inp: &[f32], att: &[f32],
                    B: usize, T: usize, C: usize, NH: usize) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    let C3 = C * 3;
    let hs = C / NH; // head size
    let scale = 1.0f32 / (hs as f32).sqrt();

    for b in 0..B {
        for t in 0..T {
            for h in 0..NH {
                let att_bth = &att[b*NH*T*T + h*T*T + t*T .. ];
                let datt_bth = &mut datt[b*NH*T*T + h*T*T + t*T .. ];
                let dpreatt_bth = &mut dpreatt[b*NH*T*T + h*T*T + t*T .. ];
                
                let query_t = &inp[b*T*C3 + t*C3 + h*hs .. ];

                // backward pass 4, through the value accumulation
                let dout_bth = &dout[b*T*C + t*C + h*hs .. ];
                for t2 in 0..=t {
                    let value_t2 = &inp[b*T*C3 + t2*C3 + h*hs + C*2 .. ];
                    let dvalue_t2 = &mut dinp[b*T*C3 + t2*C3 + h*hs + C*2 .. ];
                    for i in 0..hs {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for t2 in 0..=t {
                    for t3 in 0..=t {
                        let indicator = if t2 == t3 { 1.0f32 } else { 0.0f32 };
                        let local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                // Rust impl: Split the loops to avoid taking 2 mut ref on dinp
                let dquery_t = &mut dinp[b * T * C3 + t * C3 + h * hs .. ];
                for t2 in 0..=t {
                    let key_t2 = &inp[b * T * C3 + t2 * C3 + h * hs + C .. ];
                    for i in 0..hs {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                    }
                }

                for t2 in 0..=t {
                    let dkey_t2 = &mut dinp[b * T * C3 + t2 * C3 + h * hs + C .. ];
                    for i in 0..hs {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

#[allow(non_snake_case)]
fn residual_forward(out: &mut [f32], inp1: &[f32], inp2: &[f32], N: usize) {
    for i in 0..N {
        out[i] = inp1[i] + inp2[i];
    }
}

#[allow(non_snake_case)]
fn residual_backward(dinp1: &mut[f32], dinp2: &mut[f32], dout: &[f32], N: usize) {
    for i in 0..N {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}


#[allow(non_snake_case)]
fn gelu_forward(out: &mut [f32], inp: &[f32], N: usize) {
    let gelu_scaling_factor = (2.0f32 / std::f32::consts::PI).sqrt();
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    for i in 0..N {
        let x = inp[i];
        let  cube = 0.044715 * x * x * x;
        out[i] = 0.5 * x * (1.0 + f32::tanh(gelu_scaling_factor * (x + cube)));
    }
}

#[allow(non_snake_case)]
fn gelu_backward(dinp: &mut[f32], inp: &[f32], dout: &[f32], N: usize) {
    let gelu_scaling_factor = (2.0f32 / std::f32::consts::PI).sqrt();
    for i in 0..N {
        let x = inp[i];
        let  cube = 0.044715 * x * x * x;
        let tanh_arg = gelu_scaling_factor * (x + cube);
        let tanh_out = f32::tanh(tanh_arg);
        let coshf_out = f32::cosh(tanh_arg);
        let sech_out = 1.0 / (coshf_out * coshf_out);
        let local_grad = 0.5 * (1.0 + tanh_out) + x * 0.5 * sech_out * gelu_scaling_factor * (1.0 + 3.0 * 0.044715 * x * x);
        dinp[i] += local_grad * dout[i];
    }
}

#[allow(non_snake_case)]
fn softmax_forward(probs: &mut [f32], logits: &[f32], B: usize, T: usize, V: usize, Vp: usize) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257

    for b in 0..B {
        for t in 0..T {
            // probs <- softmax(logits)
            let logits_bt = &logits[b*T*Vp + t*Vp .. ];
            let probs_bt = &mut probs[b*T*Vp + t*Vp .. ];

            // maxval is only calculated and subtracted for numerical stability
            let mut maxval = -10000.0; // TODO something better
            for i in 0..V {
                if logits_bt[i] > maxval {
                    maxval = logits_bt[i];
                }
            }
            let mut sum = 0.0;
            for i in 0..V {
                probs_bt[i] = (logits_bt[i] - maxval).exp();
                sum += probs_bt[i];
            }
            // note we only loop to V, leaving the padded dimensions
            for i in 0..V {
                probs_bt[i] /= sum;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            for i in V..Vp {
                probs_bt[i] = 0.0;
            }
        }
    }
}

#[allow(non_snake_case)]
fn crossentropy_forward(losses: &mut [f32], probs: &[f32], targets: &[i32], B: usize, T: usize, Vp: usize) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for b in 0..B {
        for t in 0..T {
            // loss = -log(probs[target])
            let prob_bt = &probs[b*T*Vp + t*Vp .. ];
            let ix = targets[b*T + t];
            losses[b*T + t] = -(prob_bt[ix as usize]).ln();
        }
    }
}

#[allow(non_snake_case)]
fn crossentropy_softmax_backward(dlogits: &mut [f32], dlosses: &[f32], probs: &[f32], targets: &[i32], B: usize, T: usize, V: usize, Vp: usize) {
    // backwards through both softmax and crossentropy
    for b in 0..B {
        for t in 0..T {
            let dlogits_bt = &mut dlogits[b*T*Vp + t*Vp .. ];
            let probs_bt = &probs[b*T*Vp + t*Vp .. ];
            let dloss = dlosses[b*T + t];
            let ix = targets[b*T + t] as usize;
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            for i in 0..V {
                let p = probs_bt[i];
                let indicator = if i == ix { 1.0} else { 0.0 };
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

fn random_u32(state: &mut u64) -> u32 {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ((*state).wrapping_mul(0x2545F4914F6CDD1D_u64) >> 32) as u32
}

fn random_f32(state: &mut u64) -> f32 { // // random float32 in [0,1)
    ((random_u32(state) >> 8) as f32) / 16777216.0f32
}

fn sample_mult(probabilities: &[f32], coin: f32) -> i32 {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    let mut cdf = 0.0f32;
    for (i, &p) in probabilities.iter().enumerate() {
        cdf += p;
        if coin < cdf {
            return i as i32;
        }
    }
    (probabilities.len() - 1) as i32// in case of rounding errors
}

// main training loop
#[allow(non_snake_case)]
fn main() {
    let mut model= GPT2::build_from_checkpoint("gpt2_124M.bin");

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    let tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    let tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    let tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    let tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";

    let train_tokens = if std::path::Path::new(tiny_shakespeare_train).exists() { tiny_shakespeare_train } else { tiny_stories_train};
    let val_tokens = if std::path::Path::new(tiny_shakespeare_val).exists() { tiny_shakespeare_val } else { tiny_stories_val};

    let B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    let T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2

    let mut train_loader = Dataloader::new(train_tokens, B, T, true);
    let mut val_loader = Dataloader::new(val_tokens, B, T, false);
    let train_num_batches = train_loader.num_tokens / (B*T);
    let val_num_batches = val_loader.num_tokens / (B*T);
    println!("train dataset num_batches: {train_num_batches}");
    println!("val dataset num_batches: {val_num_batches}");
    let val_num_batches = 5;

    let tokenizer = Tokenizer::init("gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    let mut rng_state = 1337u64;
    let mut gen_tokens = vec![0i32; B*T];
    const GEN_T: usize = 64;


    for step in 0..=40 {
        // once in a while, estimate the validation loss
        if step % 10 == 0 {
            let mut val_loss = 0.0f32;
            val_loader.reset();
            for _ in 0..val_num_batches {
                val_loader.next_batch();
                model.forward(val_loader.inputs(), Some(val_loader.targets()), B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches as f32;
            println!("val loss {}", val_loss);
        }

        // once in a while do model inference to print generated text
        if step > 0 && step % 20 == 0 {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            for i in 0..B * T {
                gen_tokens[i] = tokenizer.eot_token;
            }
            // now sample from the model autoregressively
            println!("generating:\n---\n");
            for t in 1..GEN_T {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                model.forward(&gen_tokens, None, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]

                let acts = GPT2::get_activations(model.acts_memory.as_mut().unwrap(), model.act_sizes);
                let probs_t = (t-1) * model.config.padded_vocab_size;
                let probs = &acts.probs[probs_t .. probs_t + model.config.vocab_size];
                let coin = random_f32(&mut rng_state);
                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                let next_token = sample_mult(probs, coin);
                gen_tokens[t] = next_token;

                // print the generated token, either using the Tokenizer or a fallback
                print!("{}", tokenizer.decode(next_token));
            }
            print!("\n---\n");
        }

        // do a training step
        let start = Instant::now();
        train_loader.next_batch();
        model.forward(train_loader.inputs(), Some(train_loader.targets()), B, T);
        model.zero_grad();
        model.backward();
        model.update(1e-4, 0.9, 0.999, 1e-8, 0.0, step+1);
        let end = Instant::now();
        let elapsed = end - start;
        println!("step {}: train loss {} (took {:?} ms)", step, model.mean_loss, elapsed.as_millis());        
    }


}
