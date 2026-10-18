//! Enhance a 16 kHz mono wav with Real-time RE-USE, frame by frame as a live stream would.
//!
//! cargo run --release --example enhance -- <in.wav> <out.wav> [layers] [model.safetensors]
//!
//! `layers` is 3–12 (default 8; fewer is faster). The model defaults to
//! realtime_reuse/model.safetensors.

use reuse_rs::{audio, Streaming};
use std::time::Instant;

const SAMPLE_RATE: u32 = 16000;
const N_FFT: usize = 640;
const HOP: usize = 320;

/// Clears frames where most bins are zero after decompression, as NVIDIA's inference does.
fn clear_silent_frames(mag: &mut [Vec<f32>]) {
    let bins = mag.len();
    for t in 0..mag[0].len() {
        let zeros = mag.iter().filter(|row| row[t] <= 0.0).count();
        if zeros * 2 > bins {
            mag.iter_mut().for_each(|row| row[t] = 0.0);
        }
    }
}

fn main() -> eyre::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    eyre::ensure!(
        args.len() >= 3,
        "usage: enhance <in.wav> <out.wav> [layers] [model.safetensors]"
    );
    let (input, output) = (&args[1], &args[2]);
    let layers: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(8);
    let model = args
        .get(4)
        .map(String::as_str)
        .unwrap_or("realtime_reuse/model.safetensors");

    let (wav, sr) = audio::load_wav(input)?;
    eyre::ensure!(
        sr == SAMPLE_RATE,
        "expected {SAMPLE_RATE} Hz mono, got {sr} Hz"
    );
    let seconds = wav.len() as f32 / sr as f32;

    let start = Instant::now();
    let mut stream = Streaming::load(model, layers)?;
    println!(
        "{layers} layers loaded in {:.2} s",
        start.elapsed().as_secs_f32()
    );

    let start = Instant::now();
    let (mag, pha) = audio::stft(&wav, N_FFT, HOP, N_FFT)?;
    let (bins, frames) = (mag.len(), mag[0].len());
    let mut out_mag = vec![vec![0f32; frames]; bins];
    let mut out_pha = vec![vec![0f32; frames]; bins];
    let mut slowest = 0f64;
    for t in 0..frames {
        let m: Vec<f32> = mag.iter().map(|row| row[t].ln_1p()).collect();
        let p: Vec<f32> = pha.iter().map(|row| row[t]).collect();
        let frame_start = Instant::now();
        let (em, ep) = stream.step(&m, &p);
        if t > 5 {
            slowest = slowest.max(frame_start.elapsed().as_secs_f64() * 1e3);
        }
        for k in 0..bins {
            out_mag[k][t] = em[k];
            out_pha[k][t] = ep[k];
        }
    }

    clear_silent_frames(&mut out_mag);
    for row in &mut out_mag {
        row.iter_mut().for_each(|m| *m = m.max(0.0).exp_m1());
    }
    let mut enhanced = audio::istft(&out_mag, &out_pha, N_FFT, HOP, N_FFT)?;
    enhanced.resize(wav.len(), 1e-8);

    let took = start.elapsed().as_secs_f32();
    println!("{frames} frames, slowest frame {slowest:.1} ms (budget 20 ms)");
    println!(
        "{seconds:.1} s of audio in {took:.1} s ({:.2}x real time)",
        seconds / took
    );
    audio::save_wav(output, &enhanced, sr)?;
    println!("wrote {output}");
    Ok(())
}
