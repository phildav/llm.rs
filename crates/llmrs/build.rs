use std::{env, process::Command};

fn main() {
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        // Do nothing if the feature isn't enabled.
        return;
    }

    println!("cargo:rerun-if-changed=cuda/");

    // Set up environment variables for the Makefile
    let out_dir = env::var("OUT_DIR").unwrap();
    
    // Set GPU compute capability if specified
    if let Ok(arch) = env::var("LLMRS_NVCC_ARCH") {
        unsafe { env::set_var("GPU_COMPUTE_CAPABILITY", arch); }
    } else if let Ok(arch) = env::var("LLMRS_NVCC_ARCH") {
        unsafe { env::set_var("GPU_COMPUTE_CAPABILITY", arch); }
    }

    // Set dtype defines as environment variables
    if env::var_os("CARGO_FEATURE_BF16").is_some() {
        unsafe { env::set_var("ENABLE_BF16", "1"); }
    }
    if env::var_os("CARGO_FEATURE_FP16").is_some() {
        unsafe { env::set_var("ENABLE_FP16", "1"); }
    }

    // Set the output directory for the Makefile
    unsafe { env::set_var("TARGET_DIR", &out_dir); }

    // Change to the cuda directory and run make
    let cuda_dir = std::path::Path::new("cuda");
    if !cuda_dir.exists() {
        panic!("CUDA directory not found at {:?}", cuda_dir);
    }

    // Run make to build the CUDA kernel library
    let status = Command::new("make")
        .current_dir(cuda_dir)
        .arg("-f")
        .arg("Makefile.lib")
        .arg("all")
        .status()
        .expect("Failed to execute make");

    if !status.success() {
        panic!("Make failed with exit code: {}", status);
    }

    // The library is now built directly in the target directory
    let lib_path = std::path::Path::new(&out_dir).join("libllmrs_kernels.a");
    if !lib_path.exists() {
        panic!("Library file not found: {:?}", lib_path);
    }

    // Tell Cargo about the generated files so they get cleaned with cargo clean
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=llmrs_kernels");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cublasLt");
}

