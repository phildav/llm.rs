import os
import modal

# image = "nvidia/cuda:12.4.0-devel-ubuntu22.04"
IMAGE = "nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04"

GPU_TYPE = os.environ.get("GPU_TYPE", "L4")

APP_NAME = "cuda dev"


image = (modal.Image.from_registry(IMAGE, add_python="3.11")  # adding python is required for modal to work
    .apt_install("build-essential", "curl", "libclang-dev", "clang", "llvm-dev")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")  # Install a recent version of Rust
    .env({"PATH": "/root/.cargo/bin:$PATH"}) 
    .run_commands("mkdir -p /root/llmrs")
    .workdir("/root/llmrs")
    .add_local_dir("./", remote_path="/root/llmrs", ignore=["target/", ".venv/", '.git/'])
)


app = modal.App(APP_NAME)

def execute_command(command: str, cwd=None, env=None):
    import subprocess
    command_args = command.split(" ")
    print(f"{command_args = }")
    full_env = os.environ.copy()
    if env is not None:
        full_env.update(env)
    subprocess.run(command_args, check=True, cwd=cwd, env=full_env)


@app.function(gpu=GPU_TYPE, image=image, timeout=10000)
def run_on_modal(command: str):
    print("Running on modal")
    os.environ["RUSTFLAGS"] = "-C opt-level=3 -C codegen-units=1 -C target-cpu=native -C embed-bitcode=yes"
    execute_command("ls -la")
    execute_command("cargo --version")
    execute_command("rustc --version")
    execute_command("cargo clean")
    execute_command("cargo build --release")
    execute_command(command, env={"RUST_BACKTRACE": "full"})


@app.local_entrypoint()
def main(command: str):
    run_on_modal.remote(command)