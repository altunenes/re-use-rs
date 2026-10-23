//! Enhance a whole mono wav with offline RE-USE (nvidia/NVIDIA-NemotronLabs-RE-USE).
//!
//! cargo run --release --example offline -- <in.wav> <out.wav> <model.safetensors> [chunk_seconds] [hop_fraction]
//!
//! Without `chunk_seconds` the whole file goes through the model at once. With it, the file is
//! processed in overlapping chunks (NVIDIA uses 5 and 0.5), so memory stays bounded.
//!
//! Built with a GPU feature (`metal`, `vulkan`, `wgpu`, `cuda`, `rocm`), `REUSE_GPU=<that name>`
//! runs the Mamba blocks on the GPU.

use reuse_rs::{audio, Offline};
use std::time::Instant;

#[cfg(feature = "gpu")]
fn load(path: &str) -> eyre::Result<Offline> {
    use reuse_rs::GpuDevice;
    let device = match std::env::var("REUSE_GPU").as_deref() {
        Err(_) => return Offline::load(path),
        Ok("wgpu") => GpuDevice::Wgpu,
        Ok("metal") => GpuDevice::Metal,
        Ok("vulkan") => GpuDevice::Vulkan,
        Ok("cuda") => GpuDevice::Cuda(0),
        Ok("rocm") => GpuDevice::Rocm(0),
        Ok(other) => return Err(eyre::eyre!("unknown REUSE_GPU {other}")),
    };
    println!("GPU: {device:?}");
    Offline::load_gpu(path, device)
}

#[cfg(not(feature = "gpu"))]
fn load(path: &str) -> eyre::Result<Offline> {
    Offline::load(path)
}

fn main() -> eyre::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    eyre::ensure!(
        (4..=6).contains(&args.len()),
        "usage: offline <in.wav> <out.wav> <model.safetensors> [chunk_seconds] [hop_fraction]"
    );
    let chunk: Option<f32> = args.get(4).map(|v| v.parse()).transpose()?;
    let hop: f32 = args.get(5).map(|v| v.parse()).transpose()?.unwrap_or(0.5);
    let (wav, sample_rate) = audio::load_wav(&args[1])?;
    let seconds = wav.len() as f32 / sample_rate as f32;

    let start = Instant::now();
    let model = load(&args[3])?;
    println!("loaded in {:.2} s", start.elapsed().as_secs_f32());
    if std::env::var("REUSE_GPU").is_ok() {
        // GPU kernels compile on first use; keep that out of the timing.
        model.enhance(&wav[..wav.len().min(sample_rate as usize)], sample_rate)?;
    }

    let start = Instant::now();
    let enhanced = match chunk {
        Some(chunk) => model.enhance_chunked(&wav, sample_rate, chunk, hop)?,
        None => model.enhance(&wav, sample_rate)?,
    };
    let took = start.elapsed().as_secs_f32();
    println!(
        "{seconds:.1} s of {sample_rate} Hz audio in {took:.1} s ({:.2}x real time)",
        seconds / took
    );
    audio::save_wav(&args[2], &enhanced, sample_rate)
}
