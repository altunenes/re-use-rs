//! Enhance a mono wav with Real-time RE-USE, fed in small chunks as a live stream would.
//!
//! cargo run --release --example enhance -- <in.wav> <out.wav> [layers] [look_ahead] [model.safetensors]
//!
//! The wav can be 8, 16, 22.05, 24, 32, 44.1 or 48 kHz. `layers` is 3–12 (default 8; fewer is
//! faster), `look_ahead` is 0–2 (default 0; each frame adds 20 ms of latency). The model defaults
//! to realtime_reuse/model.safetensors.

use reuse_rs::{audio, Config, Denoiser};
use std::time::Instant;

/// Samples handed to the denoiser per call, like one buffer from an audio device.
const CHUNK: usize = 1000;

fn main() -> eyre::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    eyre::ensure!(
        args.len() >= 3,
        "usage: enhance <in.wav> <out.wav> [layers] [look_ahead] [model.safetensors]"
    );
    let (input, output) = (&args[1], &args[2]);
    let layers: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(8);
    let look_ahead: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(0);
    let model = args
        .get(5)
        .map(String::as_str)
        .unwrap_or("realtime_reuse/model.safetensors");

    let (wav, sample_rate) = audio::load_wav(input)?;
    let seconds = wav.len() as f32 / sample_rate as f32;

    let start = Instant::now();
    let config = Config {
        layers,
        sample_rate,
        look_ahead,
    };
    let mut denoiser = Denoiser::with_config(model, config)?;
    println!(
        "{layers} layers, look-ahead {look_ahead}, {sample_rate} Hz, loaded in {:.2} s",
        start.elapsed().as_secs_f32()
    );

    let start = Instant::now();
    let mut enhanced = Vec::with_capacity(wav.len());
    let mut slowest = 0f64;
    for chunk in wav.chunks(CHUNK) {
        let chunk_start = Instant::now();
        enhanced.extend(denoiser.process(chunk)?);
        slowest = slowest.max(chunk_start.elapsed().as_secs_f64() * 1e3);
    }
    enhanced.extend(denoiser.flush()?);

    let took = start.elapsed().as_secs_f32();
    let chunk_ms = CHUNK as f64 * 1e3 / sample_rate as f64;
    println!("slowest {CHUNK}-sample chunk {slowest:.1} ms (budget {chunk_ms:.1} ms)");
    println!(
        "{seconds:.1} s of audio in {took:.1} s ({:.2}x real time)",
        seconds / took
    );
    audio::save_wav(output, &enhanced, sample_rate)?;
    println!("wrote {output}");
    Ok(())
}
