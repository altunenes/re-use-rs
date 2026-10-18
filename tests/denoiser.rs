//! `Denoiser` streaming behaviour. Tests that need the model run only when `REUSE_MODEL` points at
//! `model.safetensors` from huggingface.co/nvidia/Real-time_RE-USE; otherwise they are skipped.

use reuse_rs::Denoiser;

const LAYERS: usize = 4;

fn denoiser() -> Option<Denoiser> {
    let Ok(path) = std::env::var("REUSE_MODEL") else {
        eprintln!("REUSE_MODEL not set, skipping");
        return None;
    };
    Some(Denoiser::load(path, LAYERS).expect("failed to load REUSE_MODEL"))
}

/// A tone with noise, deterministic.
fn signal(len: usize) -> Vec<f32> {
    let mut seed = 1u32;
    (0..len)
        .map(|i| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (seed >> 8) as f32 / (1 << 24) as f32 - 0.5;
            0.3 * (i as f32 * 0.07).sin() + 0.05 * noise
        })
        .collect()
}

fn run(denoiser: &mut Denoiser, input: &[f32], chunk: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for piece in input.chunks(chunk) {
        out.extend(denoiser.process(piece).unwrap());
    }
    out.extend(denoiser.flush().unwrap());
    out
}

#[test]
fn missing_model_is_an_error() {
    assert!(Denoiser::load("does/not/exist.safetensors", LAYERS).is_err());
}

#[test]
fn output_lags_input_by_latency() {
    let Some(mut denoiser) = denoiser() else {
        return;
    };
    let input = signal(10 * Denoiser::FRAME);
    let out = denoiser.process(&input).unwrap();
    assert_eq!(out.len(), input.len() - Denoiser::LATENCY);
    assert_eq!(out.len() % Denoiser::FRAME, 0);
}

#[test]
fn flush_returns_every_sample() {
    let Some(mut denoiser) = denoiser() else {
        return;
    };
    for len in [0, 1, 319, 320, 321, 640, 1000, 16000] {
        let out = run(&mut denoiser, &signal(len), 777);
        assert_eq!(out.len(), len, "input of {len} samples");
    }
}

#[test]
fn chunk_size_does_not_change_output() {
    let Some(mut denoiser) = denoiser() else {
        return;
    };
    let input = signal(8000);
    let whole = run(&mut denoiser, &input, input.len());
    for chunk in [1, 160, 320, 999] {
        assert_eq!(
            run(&mut denoiser, &input, chunk),
            whole,
            "chunks of {chunk}"
        );
    }
}

#[test]
fn reset_starts_a_new_stream() {
    let Some(mut denoiser) = denoiser() else {
        return;
    };
    let input = signal(4000);
    let first = run(&mut denoiser, &input, 500);
    denoiser.process(&signal(1234)).unwrap();
    denoiser.reset();
    assert_eq!(run(&mut denoiser, &input, 500), first);
}
