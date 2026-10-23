//! Speech enhancement with NVIDIA's RE-USE models, in plain Rust.
//!
//! Weights come straight from each model's `model.safetensors`.
//!
//! - Real-time RE-USE (<https://huggingface.co/nvidia/Real-time_RE-USE>) for live audio:
//!   [`Denoiser`] takes mono samples in any chunk size and returns enhanced samples;
//!   [`Streaming`] is the model itself, one 20 ms STFT frame per call.
//! - RE-USE and RE-USE-laughter (<https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE>) for
//!   finished recordings: [`Offline`], on the CPU, or with a GPU feature (`metal`, `vulkan`,
//!   `wgpu`, `cuda`, `rocm`) on the GPU.
//!
//! [`audio`] has the matching STFT/iSTFT and wav helpers.
//!
//! ```no_run
//! use reuse_rs::{Config, Denoiser};
//!
//! // 16 kHz, no look-ahead:
//! let mut denoiser = Denoiser::load("realtime_reuse/model.safetensors", 8)?;
//! # let noisy_chunk = vec![0f32; 1600];
//! let clean = denoiser.process(&noisy_chunk)?;
//! let rest = denoiser.flush()?;
//!
//! // Any supported sample rate, and up to 2 frames of look-ahead:
//! let config = Config { layers: 4, sample_rate: 48000, look_ahead: 1 };
//! let mut denoiser = Denoiser::with_config("realtime_reuse/model.safetensors", config)?;
//!
//! // A finished recording with the offline model:
//! # let samples = vec![0f32; 16000];
//! let model = reuse_rs::Offline::load("offline_reuse/model.safetensors")?;
//! let clean = model.enhance_chunked(&samples, 16000, 5.0, 0.5)?;
//! # Ok::<(), eyre::Report>(())
//! ```

#[cfg(all(
    feature = "gpu",
    not(any(
        feature = "wgpu",
        feature = "metal",
        feature = "vulkan",
        feature = "cuda",
        feature = "rocm"
    ))
))]
compile_error!(
    "the `gpu` feature needs a backend: enable `wgpu`, `metal`, `vulkan`, `cuda` or `rocm`"
);

pub mod audio;
mod denoiser;
#[cfg(feature = "gpu")]
mod gpu;
mod layers;
mod offline;
mod ops;
mod realtime;
mod weights;

pub use denoiser::{Config, Denoiser};
#[cfg(feature = "gpu")]
pub use gpu::GpuDevice;
pub use offline::Offline;
pub use realtime::Streaming;
