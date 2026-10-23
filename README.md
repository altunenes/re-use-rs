# reuse-rs

[![crates.io](https://img.shields.io/crates/v/reuse-rs.svg)](https://crates.io/crates/reuse-rs)
[![docs.rs](https://img.shields.io/docsrs/reuse-rs)](https://docs.rs/reuse-rs)
[![CI](https://github.com/altunenes/re-use-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/altunenes/re-use-rs/actions/workflows/ci.yml)

NVIDIA's RE-USE speech enhancement models (Mamba) in plain Rust, 8–48 kHz mono.

| | [Real-time RE-USE](https://huggingface.co/nvidia/Real-time_RE-USE) | [RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE), [RE-USE-laughter](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE-laughter) |
|---|---|---|
| API | `Denoiser`, `Streaming` | `Offline` |
| For | live audio, 20 ms frames, 0–2 frames look-ahead | recordings, whole or in chunks |
| Runs on | CPU | CPU, or GPU with a cargo feature |
| Apple M3, 16 kHz | 8.4–14.7 ms per frame (4–8 layers) | 0.9× real time CPU, 2.4× GPU |
| Weights license | non-commercial | OpenMDW-1.1, commercial use allowed |

## Demo

Real-time model, 8 layers, CPU:

<table>
  <tr>
    <th></th>
    <th>Noisy</th>
    <th>Enhanced</th>
  </tr>
  <tr>
    <td><b><a href="https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k">VoiceBank-DEMAND</a></b></td>
    <td><video src="https://github.com/user-attachments/assets/293ec7a3-fdee-42bb-a0eb-623ca9cae97b" controls width="320"></video></td>
    <td><video src="https://github.com/user-attachments/assets/b4cf4b52-c5ca-4aed-88c4-1f393fa6ac09" controls width="320"></video></td>
  </tr>
  <tr>
    <td><b><a href="https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k">VoiceBank-DEMAND</a></b></td>
    <td><video src="https://github.com/user-attachments/assets/cf4e3097-1d53-42a3-9453-741ba36a5402" controls width="320"></video></td>
    <td><video src="https://github.com/user-attachments/assets/a643c0af-b3ad-4ebc-9233-4949e5c79277" controls width="320"></video></td>
  </tr>
</table>

### Fewer layers

The real-time model can stop after any of 3–12 layers. On an 11.8 s VoiceBank-DEMAND recording,
4 layers is about 1.75× faster than 8, and I could not hear a difference between them (maybe you
can lol).

<table>
  <tr>
    <th>Original (clean)</th>
    <th>Noisy</th>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/d4e41ed9-7eeb-4183-bc2e-433a9baa1847" controls width="320"></video></td>
    <td><video src="https://github.com/user-attachments/assets/2e77ec05-1820-4a09-a645-bb14e9da8b91" controls width="320"></video></td>
  </tr>
  <tr>
    <th>4 layers</th>
    <th>6 layers</th>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/e0deba0e-5dc0-4cfe-b342-73fed48d4aa7" controls width="320"></video></td>
    <td><video src="https://github.com/user-attachments/assets/dedde0cc-7db0-4e57-93b1-5a9cb75aceb3" controls width="320"></video></td>
  </tr>
  <tr>
    <th>8 layers</th>
    <th></th>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/660f8ccc-dced-43d8-a6fa-85b91458cff7" controls width="320"></video></td>
    <td></td>
  </tr>
</table>

| SI-SDR vs. clean | Noisy | 4 layers | 6 layers | 8 layers |
|---|---|---|---|---|
| | 4.5 dB | 16.4 dB | 16.6 dB | 16.7 dB |

## Usage

Weights:

```sh
mkdir -p realtime_reuse offline_reuse
curl -L -o realtime_reuse/model.safetensors \
  https://huggingface.co/nvidia/Real-time_RE-USE/resolve/main/model.safetensors
curl -L -o offline_reuse/model.safetensors \
  https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/resolve/main/model.safetensors
```

For RE-USE-laughter, use `NVIDIA-NemotronLabs-RE-USE-laughter` in the second URL.

```sh
# Real-time: <in> <out> [layers 3–12] [look-ahead 0–2]
cargo run --release --example enhance -- noisy.wav clean.wav 8 0
# Offline: whole file, or 5 s chunks with half overlap
cargo run --release --example offline -- noisy.wav clean.wav offline_reuse/model.safetensors 5 0.5
# Offline, Mamba layers on the GPU (metal, vulkan, wgpu, cuda or rocm)
REUSE_GPU=metal cargo run --release --features metal --example offline -- noisy.wav clean.wav offline_reuse/model.safetensors 5 0.5
```

```rust
use reuse_rs::{Config, Denoiser, Offline};

let config = Config { layers: 4, sample_rate: 48000, look_ahead: 0 };
let mut denoiser = Denoiser::with_config("realtime_reuse/model.safetensors", config)?;
let clean = denoiser.process(&noisy_chunk)?; // any chunk size
let rest = denoiser.flush()?;

let model = Offline::load("offline_reuse/model.safetensors")?;
let clean = model.enhance_chunked(&samples, 16000, 5.0, 0.5)?;
// With a GPU feature: Offline::load_gpu(path, GpuDevice::Metal)
```

`Denoiser` output is delayed by `latency()` samples: 20 ms, plus 20 ms per look-ahead frame.
`Streaming` is the model itself, one STFT frame per call; `audio` has the STFT and wav helpers.
The GPU features pull in [Burn](https://github.com/tracel-ai/burn) and take a while to build.

## Benchmarks

Apple M3 (8 cores, 16 GB), macOS 26.5.

Real-time model, 16 kHz, 12 s of speech, idle machine. Under 20 ms per frame keeps up with a live
stream:

| Layers | reuse-rs (CPU) | ONNX Runtime (WebGPU) |
|---|---|---|
| 4 | **8.4 ms** | 16.9 ms |
| 6 | **11.5 ms** | 17.4 ms |
| 8 | **14.7 ms** | 20.6 ms |

Same model exported to ONNX (scans as `com.microsoft.LinearAttention`) through `ort`. ONNX Runtime
on CPU: about 70 ms. Peak memory: 140–165 MB vs. 410–440 MB.

Real-time model with its Mamba layers on the GPU (Burn, Metal), 5 s of speech, light background
load. The median improves, but every frame waits for the GPU, so slow frames get worse; it stays
on the CPU:

| Median / p95 per frame | CPU | GPU |
|---|---|---|
| 16 kHz, 4 layers | 7.5 / 8.1 ms | 6.7 / 13.1 ms |
| 16 kHz, 8 layers | 12.9 / 13.8 ms | 9.2 / 17.6 ms |
| 48 kHz, 4 layers | 19.6 / 21.2 ms | 17.1 / 26.2 ms |
| 48 kHz, 8 layers | 32.8 / 34.7 ms | 25.1 / 33.4 ms |

Offline model, 5.1 s recording in one pass, best of 3–5 runs, light background load. GPU: Mamba
layers on Metal, convolutions on the CPU (running those on the GPU too was slower):

| | CPU | GPU (Metal) |
|---|---|---|
| 16 kHz | 5.8 s (0.9× real time) | **2.1 s** (2.4×) |
| 48 kHz | 19.4 s (0.26×) | **6.0 s** (0.85×) |

```sh
cargo run --release --example bench -- noisy.wav 8 12
```

## How it works

- Everything is in `src/`: convolutions as im2col plus one matrix multiply (Accelerate on macOS,
  `matrixmultiply` elsewhere), the Mamba scan over fixed 16-state arrays with a polynomial `exp` so
  it vectorizes, scans across channels in parallel with `rayon`.
- Real-time: per 20 ms STFT frame, a causal conv encoder, per layer a causal time Mamba (state per
  frequency bin) and a bidirectional Mamba across frequency, then magnitude and phase decoders.
- Offline: the same blocks over the whole input. The encoder shrinks time by 4, 30 layers run
  bidirectional Mambas over time and frequency, and InstanceNorm spans the whole input, so it
  needs the recording (or a chunk) up front. On the GPU each Mamba is three matmuls and two CubeCL
  kernels (causal conv; scan with gating fused).
- Output matches float64 runs of NVIDIA's PyTorch models: 67–112 dB SNR (real-time), 77–108 dB
  (offline; about 58 dB on one loud passage, where PyTorch float32 itself gets 64 dB).

## Limitations

- Only measured on an Apple M3. Without Accelerate (`matrixmultiply`), 8 layers take 21.4 ms per
  frame on the same machine, so 4–6 layers is safer on other platforms.
- 48 kHz is about 3× the work of 16 kHz: 19.6 ms per frame with 4 layers, right at the limit. For
  live audio, resample to 16 kHz.
- The offline models are for files: not real time on the CPU, and they need the input up front.
- The GPU features build for Metal, Vulkan, wgpu, CUDA and ROCm, but have only been run on Metal.
- Some recordings are not cleaned well by the real-time model. In my tests these were mostly
  recordings where the background noise also has other people talking in it. The offline models
  clean them much better, almost completely, so use `Offline` for these:

<table>
  <tr>
    <th colspan="2">Recording 1</th>
  </tr>
  <tr>
    <th>Noisy</th>
    <th>Real-time (8 layers)</th>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/d779122f-277b-4ce6-b8d3-af157e75dfe2" controls width="320"></video></td>
    <td><video src="https://github.com/user-attachments/assets/353c9289-dab6-4b72-ab1e-6280beb9af18" controls width="320"></video></td>
  </tr>
  <tr>
    <th>RE-USE</th>
    <th>RE-USE-laughter</th>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/911944b2-8fc6-47cd-80e6-a8a8d21204a2" controls width="320"></video></td>
    <td><video src="https://github.com/user-attachments/assets/354a3095-9987-4989-803c-c66723f4bfa6" controls width="320"></video></td>
  </tr>
  <tr>
    <th colspan="2">Recording 2</th>
  </tr>
  <tr>
    <th>Noisy</th>
    <th>Real-time (8 layers)</th>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/f860026e-e383-463e-b90f-a61e95280a6a" controls width="320"></video></td>
    <td><video src="https://github.com/user-attachments/assets/ccad0efb-df9d-4bad-abb1-79ba12082eec" controls width="320"></video></td>
  </tr>
  <tr>
    <th>RE-USE</th>
    <th>RE-USE-laughter</th>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/8b4cb539-4f5b-46ec-b31c-41e1d0769a82" controls width="320"></video></td>
    <td><video src="https://github.com/user-attachments/assets/177414e9-8b8f-4ae6-9915-d3542bb38b66" controls width="320"></video></td>
  </tr>
</table>

## License

Code: MIT or Apache-2.0. Weights are NVIDIA's: Real-time RE-USE under the
[NVIDIA One-Way Noncommercial License](https://github.com/NVlabs/HMAR/blob/main/LICENSE); RE-USE and
RE-USE-laughter under OpenMDW-1.1 (see the
[model card](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE)).

- [One Model, Many Latencies: Universal Speech Enhancement for Diverse Real-Time Applications](http://arxiv.org/abs/2606.25621) (IEEE SLT 2026)
- [Rethinking Training Targets, Architectures and Data Quality for Universal Speech Enhancement](https://arxiv.org/abs/2603.02641) (NeurIPS 2026)
