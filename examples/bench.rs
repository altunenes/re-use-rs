//! Per-frame speed of Real-time RE-USE on the start of a 16 kHz wav.
//!
//! cargo run --release --example bench -- <in.wav> [layers] [seconds] [model.safetensors]
//!   e.g. -- noisy.wav 8 12
//!
//! Each frame is 20 ms of audio, so a median under 20 ms keeps up with a live stream.

use reuse_rs::{Streaming, audio};
use std::time::Instant;

fn main() -> eyre::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    eyre::ensure!(args.len() >= 2, "usage: bench <in.wav> [layers] [seconds] [model.safetensors]");
    let input = &args[1];
    let layers: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(8);
    let seconds: f32 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(12.0);
    let model = args.get(4).map(String::as_str).unwrap_or("realtime_reuse/model.safetensors");

    let (wav, sr) = audio::load_wav(input)?;
    eyre::ensure!(sr == 16000, "expected 16 kHz audio, got {sr}");
    let wav = &wav[..((seconds * sr as f32) as usize).min(wav.len())];
    let (mag, pha) = audio::stft(wav, 640, 320, 640)?;
    let frames = mag[0].len();

    let mut stream = Streaming::load(model, layers)?;
    let mut times = Vec::with_capacity(frames);
    for t in 0..frames {
        let m: Vec<f32> = mag.iter().map(|row| row[t].ln_1p()).collect();
        let p: Vec<f32> = pha.iter().map(|row| row[t]).collect();
        let start = Instant::now();
        std::hint::black_box(stream.step(&m, &p));
        times.push(start.elapsed().as_secs_f64() * 1e3);
    }

    let mut warm = times[5.min(times.len())..].to_vec();
    warm.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = warm[warm.len() / 2];
    println!(
        "{layers} layers, {frames} frames: median {median:.2} ms, p95 {:.2} ms, max {:.2} ms -> {:.2}x real time",
        warm[warm.len() * 95 / 100],
        warm[warm.len() - 1],
        20.0 / median
    );
    Ok(())
}
