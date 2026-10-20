//! `Denoiser` streaming behaviour. Tests that need the model run only when `REUSE_MODEL` points at
//! `model.safetensors` from huggingface.co/nvidia/Real-time_RE-USE; otherwise they are skipped.

use reuse_rs::{audio, Config, Denoiser};

const LAYERS: usize = 4;

fn model() -> Option<String> {
    let path = std::env::var("REUSE_MODEL").ok();
    if path.is_none() {
        eprintln!("REUSE_MODEL not set, skipping");
    }
    path
}

fn denoiser(path: &str, sample_rate: u32, look_ahead: usize) -> Denoiser {
    let config = Config {
        layers: LAYERS,
        sample_rate,
        look_ahead,
    };
    Denoiser::with_config(path, config).expect("failed to load REUSE_MODEL")
}

/// Every supported sample rate with every look-ahead.
fn configs() -> impl Iterator<Item = (u32, usize)> {
    audio::SAMPLE_RATES
        .into_iter()
        .flat_map(|sr| (0..=2).map(move |la| (sr, la)))
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
fn unsupported_settings_are_errors() {
    assert!(audio::stft_sizes(11025).is_err());
    let Some(path) = model() else {
        return;
    };
    let bad = [
        Config {
            sample_rate: 11025,
            ..Config::default()
        },
        Config {
            look_ahead: 3,
            ..Config::default()
        },
        Config {
            layers: 13,
            ..Config::default()
        },
    ];
    for config in bad {
        assert!(Denoiser::with_config(&path, config).is_err(), "{config:?}");
    }
}

#[test]
fn stft_sizes_match_nvidia() {
    let sizes = audio::SAMPLE_RATES.map(|sr| audio::stft_sizes(sr).unwrap());
    assert_eq!(
        sizes,
        [
            (320, 160),
            (640, 320),
            (882, 442),
            (960, 480),
            (1280, 640),
            (1764, 882),
            (1920, 960)
        ]
    );
}

#[test]
fn output_lags_input_by_latency() {
    let Some(path) = model() else {
        return;
    };
    for (sr, la) in configs() {
        let mut d = denoiser(&path, sr, la);
        assert_eq!(d.frame(), audio::stft_sizes(sr).unwrap().1);
        let input = signal(10 * d.frame());
        let out = d.process(&input).unwrap();
        // Output comes in whole frames, so it can trail by up to one frame more.
        let ready = input.len() - d.latency();
        assert!(
            out.len() <= ready && out.len() + d.frame() > ready,
            "{sr} Hz, look-ahead {la}"
        );
    }
}

#[test]
fn flush_returns_every_sample() {
    let Some(path) = model() else {
        return;
    };
    for (sr, la) in configs() {
        let mut d = denoiser(&path, sr, la);
        for len in [
            0,
            1,
            d.frame() - 1,
            d.frame(),
            d.frame() + 1,
            1000,
            sr as usize / 5,
        ] {
            let out = run(&mut d, &signal(len), 777);
            assert_eq!(out.len(), len, "{sr} Hz, look-ahead {la}, {len} samples");
        }
    }
}

#[test]
fn chunk_size_does_not_change_output() {
    let Some(path) = model() else {
        return;
    };
    for (sr, la) in configs() {
        let mut d = denoiser(&path, sr, la);
        let input = signal(sr as usize / 5);
        let whole = run(&mut d, &input, input.len());
        for chunk in [1, 160, 999] {
            assert_eq!(
                run(&mut d, &input, chunk),
                whole,
                "{sr} Hz, look-ahead {la}, chunks of {chunk}"
            );
        }
    }
}

#[test]
fn reset_starts_a_new_stream() {
    let Some(path) = model() else {
        return;
    };
    for (sr, la) in configs() {
        let mut d = denoiser(&path, sr, la);
        let input = signal(4000);
        let first = run(&mut d, &input, 500);
        d.process(&signal(1234)).unwrap();
        d.reset();
        assert_eq!(run(&mut d, &input, 500), first, "{sr} Hz, look-ahead {la}");
    }
}
