use std::{env};

fn arch_flags() -> Vec<String> {
    if let Ok(s) = env::var("LLMRS_NVCC_DEV_ARCH") { s.split_whitespace().map(|s| s.to_string()).collect() }
    else if let Ok(s) = env::var("LLMRS_NVCC_ARCH") { s.split_whitespace().map(|s| s.to_string()).collect() }
    else { vec!["-gencode=arch=compute_86,code=sm_86".into()] }
}

fn main() {

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        // Do nothing if the feature isn't enabled.
        return;
    }

    println!("cargo:rerun-if-changed=cuda/");

    let mut b = cc::Build::new();
    b.cuda(true)
        .flag("-O3")
        .flag("-std=c++14")
        .flag("--use_fast_math")  // Add fast math flag as shown in .cu file
        .include("cuda");         // Add include path for common.h

    // dtype defines to control floatX inside the .cu
    if env::var_os("CARGO_FEATURE_BF16").is_some() { b.define("ENABLE_BF16", None); }
    if env::var_os("CARGO_FEATURE_FP16").is_some() { b.define("ENABLE_FP16", None); }

    for f in arch_flags() { b.flag(&f); }
    
    // Build all .cu files in the cuda directory
    let cuda_dir = std::path::Path::new("cuda");
    if let Ok(entries) = std::fs::read_dir(cuda_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                if let Some(extension) = entry.path().extension() {
                    if extension == "cu" {
                        println!("Building CUDA file: {}", entry.path().display());
                        b.file(entry.path());
                    }
                }
            }
        }
    }
    
    b.compile("llmrs_dev_kernels");

    // Explicitly tell Cargo to link against our compiled library
    println!("cargo:rustc-link-search=native={}", std::env::var("OUT_DIR").unwrap());
    println!("cargo:rustc-link-lib=static=llmrs_dev_kernels");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cublasLt");

    //if let Ok(cuda) = env::var("CUDA_HOME").or_else(|_| env::var("CUDA_PATH")) {
    //    println!("cargo:rustc-link-search=native={}/lib64", cuda);
    //}
}

