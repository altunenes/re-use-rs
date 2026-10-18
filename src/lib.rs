//! Speech enhancement with NVIDIA's Real-time RE-USE model, in plain Rust on the CPU.
//!
//! Weights come straight from the model's `model.safetensors`
//! (<https://huggingface.co/nvidia/Real-time_RE-USE>). [`Streaming`] enhances one 20 ms STFT
//! frame per call; [`audio`] has the matching STFT/iSTFT and wav helpers.
//!
//! ```no_run
//! use reuse_rs::{audio, Streaming};
//!
//! let mut model = Streaming::load("realtime_reuse/model.safetensors", 8)?;
//! let (wav, _) = audio::load_wav("noisy.wav")?;
//! let (mag, pha) = audio::stft(&wav, 640, 320, 640)?;
//! for t in 0..mag[0].len() {
//!     let m: Vec<f32> = mag.iter().map(|row| row[t].ln_1p()).collect();
//!     let p: Vec<f32> = pha.iter().map(|row| row[t]).collect();
//!     let (clean_mag, clean_pha) = model.step(&m, &p);
//! }
//! # Ok::<(), eyre::Report>(())
//! ```

pub mod audio;
mod layers;
mod ops;
mod realtime;
mod weights;

pub use realtime::Streaming;
