# reuse-rs

[![crates.io](https://img.shields.io/crates/v/reuse-rs.svg)](https://crates.io/crates/reuse-rs)
[![docs.rs](https://img.shields.io/docsrs/reuse-rs)](https://docs.rs/reuse-rs)
[![CI](https://github.com/altunenes/re-use-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/altunenes/re-use-rs/actions/workflows/ci.yml)


Real-time speech enhancement in plain Rust, running NVIDIA's
[Real-time RE-USE](https://huggingface.co/nvidia/Real-time_RE-USE) model on the CPU.

- Streams one 20 ms frame at a time, faster than real time on a laptop CPU (8.4–14.7 ms per frame on an Apple M3).
- No ONNX Runtime, no GPU, no Python. The weights load straight from NVIDIA's `model.safetensors`.
- About 150 MB of memory.
- Output matches a float64 PyTorch reference to 1.5e-6, as close as PyTorch's own float32 run.

## Demo

The same recordings before and after reuse-rs (8 layers, CPU, streaming). The spectrogram fills in as the audio plays.

<table>
  <tr>
    <th></th>
    <th>Noisy</th>
    <th>Enhanced (reuse-rs, 8 layers, CPU)</th>
  </tr>
  <tr>
    <td><b>Edinburgh</b><br>2.3 s</td>
    <td><video src="https://github.com/user-attachments/assets/f559d4c9-0bf3-41de-b04f-b34af7d2cc2e" controls width="320"></video></td>
    <td><video src="https://github.com/user-attachments/assets/4afca7ce-57d5-4e0b-b0d8-df6addc5a629" controls width="320"></video></td>
  </tr>
  <tr>
    <td><b>Heavy city noise</b><br>10 s</td>
    <td><video src="https://github.com/user-attachments/assets/0a73e729-f088-420d-8108-399023dd1639" controls width="320"></video></td>
    <td><video src="https://github.com/user-attachments/assets/221564f6-6bcc-4c82-9e41-726d1dd076c6" controls width="320"></video></td>
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

The heavy city noise recording with 4, 6 and 8 layers. 4 layers is about 1.75× faster than 8 (see
[Benchmarks](#benchmarks)), and I could not hear a difference between them (maybe you can lol).

<table>
  <tr>
    <th>Noisy</th>
    <th>4 layers</th>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/0a73e729-f088-420d-8108-399023dd1639" controls width="320"></video></td>
    <td><video src="https://github.com/user-attachments/assets/c76cc9f1-d124-4517-92aa-39d354625707" controls width="320"></video></td>
  </tr>
  <tr>
    <th>6 layers</th>
    <th>8 layers</th>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/66f7025e-5f62-447c-9126-2504c98dd62c" controls width="320"></video></td>
    <td><video src="https://github.com/user-attachments/assets/956a90b0-fab8-4e55-808e-5160e3b39f67" controls width="320"></video></td>
  </tr>
</table>

But in math perspective, the outputs are close when measured too. Subtracting one output from the other leaves a difference this far below the enhanced speech:

| | Whole signal | Below 1 kHz | 1–4 kHz | Above 4 kHz |
|---|---|---|---|---|
| 8 vs 4 layers | 19.2 dB | 19.2 dB | 19.0 dB | 19.3 dB |
| 8 vs 6 layers | 18.6 dB | 18.4 dB | 20.3 dB | 21.7 dB |
| 6 vs 4 layers | 22.8 dB | 23.0 dB | 21.2 dB | 21.0 dB |

19 dB down is about 1% of the energy. There is no clean original of this recording, so this shows
how much the outputs differ, not which one is cleaner.

## Setup

Download the weights (54 MB):

```sh
mkdir -p realtime_reuse
curl -L -o realtime_reuse/model.safetensors \
  https://huggingface.co/nvidia/Real-time_RE-USE/resolve/main/model.safetensors
```

Enhance a 16 kHz mono wav:

```sh
cargo run --release --example enhance -- noisy.wav clean.wav 8
```

The last argument is the number of Mamba layers, 3–12. Fewer layers is faster (see
[Fewer layers](#fewer-layers)).
Convert other audio with e.g. `ffmpeg -i in.mp4 -ac 1 -ar 16000 noisy.wav`.

## Library

```rust
use reuse_rs::Denoiser;

let mut denoiser = Denoiser::load("realtime_reuse/model.safetensors", 8)?;
// Feed 16 kHz mono samples in whatever chunks they arrive.
let clean = denoiser.process(&noisy_chunk)?;
// At the end of the stream:
let rest = denoiser.flush()?;
```

The second argument of `load` is the number of Mamba layers (3–12). Every frame passes through each
layer, so fewer layers is faster: on an Apple M3, 4 layers take 8.4 ms per frame and 8 layers take
14.7 ms. The model was trained to stop after any of these layer counts, and the output changes
little between them (see [Fewer layers](#fewer-layers)), so 4 layers is a good choice when speed
matters.

`process` returns the enhanced samples that are ready, 320 at a time (20 ms), delayed by
`Denoiser::LATENCY` (320 samples). For frame-level control, `Streaming` takes one STFT frame
(log1p magnitude and phase, 321 bins) per call, and `audio` has the matching STFT and wav helpers.

## Benchmarks

Apple M3 (8 cores, 16 GB), macOS 26.5, idle machine, 16 kHz mono.

### Per frame

12 s of speech, look-ahead 0. A frame is 20 ms of audio, so under 20 ms per frame keeps up with a live stream.

| Layers | reuse-rs (CPU) | ONNX Runtime (WebGPU) |
|---|---|---|
| 4 | **8.4 ms** (2.4× real time) | 16.9 ms (1.2×) |
| 6 | **11.5 ms** (1.7×) | 17.4 ms (1.15×) |
| 8 | **14.7 ms** (1.4×) | 20.6 ms (0.97×) |

The ONNX Runtime column is the same model exported to ONNX (Mamba scans as
`com.microsoft.LinearAttention`) and run through the `ort` crate with WebGPU. On CPU, ONNX Runtime
took about 70 ms per frame.

### Real recordings

Time to enhance a whole recording (STFT, model, inverse STFT). Less time is faster. In brackets is
the speed compared to real time: 2× means 10 s of audio is done in 5 s. Below 1× is slower than
real time, so it cannot keep up with a live stream.

| Audio length | 4 layers | 6 layers | 8 layers | ONNX Runtime WebGPU, 8 layers |
|---|---|---|---|---|
| 2.3 s | **1.0 s** (2.2×) | 1.4 s (1.6×) | 1.8 s (1.3×) | 2.7 s (0.85×) |
| 10 s | **4.2 s** (2.4×) | 5.7 s (1.75×) | 7.1 s (1.4×) | 10.4 s (0.96×) |
| 40.7 s | **17.0 s** (2.4×) | 23.3 s (1.75×) | 30.1 s (1.35×) | 41.5 s (0.98×) |
| Peak memory | 140–150 MB | 145–155 MB | 150–165 MB | 410–440 MB |

NVIDIA's offline model (`nvidia/RE-USE`, 30 bidirectional layers) was also tried through ONNX
Runtime. It is NVIDIA's highest-quality model, but ran at 0.19–0.33× real time on WebGPU (0.10× on CPU) with
3.3–3.9 GB of memory, so it is not included.

Run the benchmark yourself:

```sh
cargo run --release --example bench -- noisy.wav 8 12
```

## How it works

Every 20 ms the input gets one STFT frame (n_fft 640, hop 320 at 16 kHz). The model is NVIDIA's
streaming SEMamba: a causal dense conv encoder, then per layer a causal time Mamba (one state
update per frequency bin) and a bidirectional Mamba across frequency, then magnitude and phase
decoders. All of it is written by hand in `src/`:

- Convolutions are im2col plus one matrix multiply (Accelerate on macOS, `matrixmultiply` elsewhere).
- The Mamba selective scan uses fixed 16-state arrays and a polynomial `exp`, so the compiler vectorizes it.
- Scans run across channels in parallel with `rayon`.

## Limitations

- Only 16 kHz input and look-ahead 0 have been tested. NVIDIA's model supports 8–48 kHz and up to
  2 frames of look-ahead.
- Speed has only been measured on an Apple M3. Other platforms use the `matrixmultiply` crate instead
  of Accelerate: on the same M3 that path takes 21.4 ms per frame at 8 layers, so 4–6 layers is the
  safer choice there.
- Some recordings are not cleaned well. In my tests these were mostly recordings where the
  background noise also has other people talking in it:

<table>
  <tr>
    <th>Noisy</th>
    <th>Enhanced (reuse-rs, 8 layers, CPU)</th>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/d779122f-277b-4ce6-b8d3-af157e75dfe2" controls width="320"></video></td>
    <td><video src="https://github.com/user-attachments/assets/353c9289-dab6-4b72-ab1e-6280beb9af18" controls width="320"></video></td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/f860026e-e383-463e-b90f-a61e95280a6a" controls width="320"></video></td>
    <td><video src="https://github.com/user-attachments/assets/ccad0efb-df9d-4bad-abb1-79ba12082eec" controls width="320"></video></td>
  </tr>
</table>

If you don't need real time, for example when cleaning finished recordings, try NVIDIA's offline re-use
model instead. see: [SpeechLens](https://github.com/faraday/SpeechLens) 

## License

The code is MIT or Apache-2.0. The model weights are NVIDIA's, under the
[NVIDIA One-Way Noncommercial License](https://github.com/NVlabs/HMAR/blob/main/LICENSE): research and
non-commercial use only.

Model papers:

- [One Model, Many Latencies: Universal Speech Enhancement for Diverse Real-Time Applications](http://arxiv.org/abs/2606.25621) (IEEE SLT 2026)
- [Rethinking Training Targets, Architectures and Data Quality for Universal Speech Enhancement](https://arxiv.org/abs/2603.02641) (NeurIPS 2026)
