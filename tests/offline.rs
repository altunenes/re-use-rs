//! `Offline` behaviour. Tests that need the model run only when `REUSE_OFFLINE_MODEL` points at
//! `model.safetensors` from huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE; otherwise they are
//! skipped.

use reuse_rs::{audio, Offline};

fn model() -> Option<Offline> {
    let Ok(path) = std::env::var("REUSE_OFFLINE_MODEL") else {
        eprintln!("REUSE_OFFLINE_MODEL not set, skipping");
        return None;
    };
    Some(Offline::load(path).expect("failed to load REUSE_OFFLINE_MODEL"))
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

#[test]
fn missing_model_is_an_error() {
    assert!(Offline::load("does/not/exist.safetensors").is_err());
}

#[test]
fn stft_sizes_match_nvidia() {
    let sizes = audio::SAMPLE_RATES.map(Offline::stft_sizes);
    assert_eq!(
        sizes,
        [
            (320, 40),
            (640, 80),
            (882, 110),
            (960, 120),
            (1280, 160),
            (1764, 220),
            (1920, 240)
        ]
    );
}

#[test]
fn output_has_input_length() {
    let Some(model) = model() else {
        return;
    };
    for sr in audio::SAMPLE_RATES {
        for len in [sr as usize / 10, sr as usize / 4 + 7] {
            let out = model.enhance(&signal(len), sr).unwrap();
            assert_eq!(out.len(), len, "{sr} Hz, {len} samples");
            assert!(out.iter().all(|v| v.is_finite()), "{sr} Hz, {len} samples");
        }
    }
    assert!(model.enhance(&signal(1000), 11025).is_err());
    assert!(model.enhance(&[], 16000).unwrap().is_empty());
    assert!(model
        .enhance_chunked(&[], 16000, 5.0, 0.5)
        .unwrap()
        .is_empty());
}

#[test]
fn chunked_covers_every_sample() {
    let Some(model) = model() else {
        return;
    };
    let input = signal(16000);
    // 0.25 s chunks at half overlap: 7 chunks, the last one aligned to the end.
    let out = model.enhance_chunked(&input, 16000, 0.25, 0.5).unwrap();
    assert_eq!(out.len(), input.len());
    assert!(out.iter().all(|v| v.is_finite()));
    // One chunk longer than the input is the same as the whole-file pass.
    let whole = model.enhance(&input, 16000).unwrap();
    let single = model.enhance_chunked(&input, 16000, 2.0, 0.5).unwrap();
    let worst = whole
        .iter()
        .zip(&single)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(worst < 1e-6, "max difference {worst}");
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_matches_cpu() {
    use reuse_rs::GpuDevice;
    let Some(cpu) = model() else {
        return;
    };
    let device = match std::env::var("REUSE_GPU").as_deref() {
        Ok("wgpu") => GpuDevice::Wgpu,
        Ok("metal") => GpuDevice::Metal,
        Ok("vulkan") => GpuDevice::Vulkan,
        Ok("cuda") => GpuDevice::Cuda(0),
        Ok("rocm") => GpuDevice::Rocm(0),
        _ => {
            eprintln!("REUSE_GPU not set, skipping");
            return;
        }
    };
    let gpu = Offline::load_gpu(std::env::var("REUSE_OFFLINE_MODEL").unwrap(), device).unwrap();
    // 3 s at 48 kHz needs more than 65535 GPU work groups in one dimension.
    for (sr, seconds) in [(8000, 0.25), (16000, 0.25), (48000, 3.0)] {
        let input = signal((sr as f32 * seconds) as usize);
        let a = cpu.enhance(&input, sr).unwrap();
        let b = gpu.enhance(&input, sr).unwrap();
        let worst = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst < 1e-4, "{sr} Hz: max difference {worst}");
    }
}
