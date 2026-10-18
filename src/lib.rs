//! Speech enhancement with NVIDIA's Real-time RE-USE model, in plain Rust on the CPU.
//!
//! Weights come straight from the model's `model.safetensors`
//! (<https://huggingface.co/nvidia/Real-time_RE-USE>). [`Denoiser`] takes 16 kHz mono samples in
//! any chunk size and returns enhanced samples; [`Streaming`] is the model itself, one 20 ms STFT
//! frame per call; [`audio`] has the matching STFT/iSTFT and wav helpers.
//!
//! ```no_run
//! use reuse_rs::Denoiser;
//!
//! let mut denoiser = Denoiser::load("realtime_reuse/model.safetensors", 8)?;
//! # let noisy_chunk = vec![0f32; 1600];
//! let clean = denoiser.process(&noisy_chunk)?;
//! let rest = denoiser.flush()?;
//! # Ok::<(), eyre::Report>(())
//! ```

pub mod audio;
mod denoiser;
mod layers;
mod ops;
mod realtime;
mod weights;

pub use denoiser::Denoiser;
pub use realtime::Streaming;
