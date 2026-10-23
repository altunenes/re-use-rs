//! Offline RE-USE (nvidia/NVIDIA-NemotronLabs-RE-USE and its laughter variant): 30
//! bidirectional TF-Mamba blocks over a whole recording.
//!
//! Conv activations are channel-major `[channels][time][freq]`; Mamba activations are
//! `[sequences][positions][channels]`.

use crate::audio::{hann_window, istft, stft, SAMPLE_RATES};
use crate::layers::BiMambaBlock;
use crate::ops::{gemm, transpose};
use crate::weights::Weights;
use eyre::Result;
use rayon::prelude::*;
use std::path::Path;

/// Conv2d over a whole `[cin][time][freq]` input.
struct Conv {
    weight: Vec<f32>,
    bias: Vec<f32>,
    cin: usize,
    cout: usize,
    kt: usize,
    kf: usize,
    dilation: usize,
    pad: (usize, usize),
    stride: (usize, usize),
}

impl Conv {
    /// `prefix.weight` is `[cout][cin][kt][kf]`.
    fn load(
        w: &Weights,
        prefix: &str,
        dilation: usize,
        pad: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        let shape = w.shape(&format!("{prefix}.weight"))?.to_vec();
        Ok(Self {
            weight: w.get(&format!("{prefix}.weight"))?,
            bias: w.get(&format!("{prefix}.bias"))?,
            cout: shape[0],
            cin: shape[1],
            kt: shape[2],
            kf: shape[3],
            dilation,
            pad,
            stride,
        })
    }

    /// Same as `load` for a `[cout][cin][1][k]` kernel applied along time instead of frequency.
    fn load_along_time(w: &Weights, prefix: &str) -> Result<Self> {
        let mut conv = Self::load(w, prefix, 1, (1, 0), (1, 1))?;
        (conv.kt, conv.kf) = (conv.kf, conv.kt);
        Ok(conv)
    }

    fn out_size(&self, time: usize, freq: usize) -> (usize, usize) {
        let span = self.dilation * (self.kt - 1) + 1;
        (
            (time + 2 * self.pad.0 - span) / self.stride.0 + 1,
            (freq + 2 * self.pad.1 - self.kf) / self.stride.1 + 1,
        )
    }

    /// `x` is `[cin][time][freq]`; returns `[cout][time_out][freq_out]`. Output time rows are
    /// computed in tiles (im2col + one GEMM each) so the column buffer stays small.
    fn forward(&self, x: &[f32], time: usize, freq: usize) -> Vec<f32> {
        let (to, fo) = self.out_size(time, freq);
        let k = self.cin * self.kt * self.kf;
        let tile = (16384 / fo).clamp(1, to);
        let tiles: Vec<Vec<f32>> = (0..to.div_ceil(tile))
            .into_par_iter()
            .map(|i| {
                let rows = (i * tile..((i + 1) * tile).min(to)).collect::<Vec<_>>();
                let n = rows.len() * fo;
                let mut cols = vec![0f32; k * n];
                for ci in 0..self.cin {
                    let plane = &x[ci * time * freq..][..time * freq];
                    for a in 0..self.kt {
                        for b in 0..self.kf {
                            let row = &mut cols[((ci * self.kt + a) * self.kf + b) * n..][..n];
                            for (j, &t) in rows.iter().enumerate() {
                                let ti = (t * self.stride.0 + a * self.dilation) as isize
                                    - self.pad.0 as isize;
                                if ti < 0 || ti as usize >= time {
                                    continue;
                                }
                                let src = &plane[ti as usize * freq..][..freq];
                                for f in 0..fo {
                                    let fi = (f * self.stride.1 + b) as isize - self.pad.1 as isize;
                                    if fi >= 0 && (fi as usize) < freq {
                                        row[j * fo + f] = src[fi as usize];
                                    }
                                }
                            }
                        }
                    }
                }
                let mut out = vec![0f32; self.cout * n];
                gemm(
                    false,
                    false,
                    self.cout,
                    n,
                    k,
                    &self.weight,
                    k,
                    &cols,
                    n,
                    0.0,
                    &mut out,
                    n,
                );
                out
            })
            .collect();
        let mut out = vec![0f32; self.cout * to * fo];
        for (i, t) in tiles.iter().enumerate() {
            let n = t.len() / self.cout;
            for co in 0..self.cout {
                let dst = &mut out[co * to * fo + i * tile * fo..][..n];
                dst.copy_from_slice(&t[co * n..][..n]);
                dst.iter_mut().for_each(|v| *v += self.bias[co]);
            }
        }
        out
    }
}

/// InstanceNorm2d (affine, eps 1e-5) followed by PReLU, per channel over the whole plane.
struct NormAct {
    weight: Vec<f32>,
    bias: Vec<f32>,
    prelu: Vec<f32>,
}

impl NormAct {
    fn load(w: &Weights, norm: &str, prelu: &str) -> Result<Self> {
        Ok(Self {
            weight: w.get(&format!("{norm}.weight"))?,
            bias: w.get(&format!("{norm}.bias"))?,
            prelu: w.get(&format!("{prelu}.weight"))?,
        })
    }

    fn apply(&self, x: &mut [f32]) {
        let plane = x.len() / self.weight.len();
        x.par_chunks_mut(plane).enumerate().for_each(|(c, row)| {
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / plane as f64;
            let var = row
                .iter()
                .map(|&v| (v as f64 - mean) * (v as f64 - mean))
                .sum::<f64>()
                / plane as f64;
            let scale = self.weight[c] as f64 / (var + 1e-5).sqrt();
            let shift = self.bias[c] as f64 - mean * scale;
            let (scale, shift, a) = (scale as f32, shift as f32, self.prelu[c]);
            for v in row {
                let y = *v * scale + shift;
                *v = if y > 0.0 { y } else { a * y };
            }
        });
    }
}

/// A conv followed by its InstanceNorm and PReLU (`prefix.0`, `prefix.1`, `prefix.2`).
struct ConvBlock {
    conv: Conv,
    act: NormAct,
}

impl ConvBlock {
    fn load(w: &Weights, prefix: &str, conv: Conv) -> Result<Self> {
        Ok(Self {
            conv,
            act: NormAct::load(w, &format!("{prefix}.1"), &format!("{prefix}.2"))?,
        })
    }

    fn forward(&self, x: &[f32], time: usize, freq: usize) -> (Vec<f32>, usize, usize) {
        let (to, fo) = self.conv.out_size(time, freq);
        let mut y = self.conv.forward(x, time, freq);
        self.act.apply(&mut y);
        (y, to, fo)
    }
}

/// DenseBlock: 4 dilated 3×3 convs, each seeing all earlier outputs and the input.
struct DenseBlock {
    layers: Vec<ConvBlock>,
}

impl DenseBlock {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        let layers = (0..4)
            .map(|i| {
                let p = format!("{prefix}.dense_block.{i}");
                let dil = 1 << i;
                let conv = Conv::load(w, &format!("{p}.0"), dil, (dil, 1), (1, 1))?;
                ConvBlock::load(w, &p, conv)
            })
            .collect::<Result<_>>()?;
        Ok(Self { layers })
    }

    /// Layer `i` reads `[y_{i-1}, …, y_0, x]`, kept as one contiguous tail of `buf`.
    fn forward(&self, x: &[f32], time: usize, freq: usize) -> Vec<f32> {
        let size = x.len();
        let depth = self.layers.len();
        let mut buf = vec![0f32; (depth + 1) * size];
        buf[depth * size..].copy_from_slice(x);
        let mut y = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            y = layer.forward(&buf[(depth - i) * size..], time, freq).0;
            if i + 1 < depth {
                buf[(depth - 1 - i) * size..(depth - i) * size].copy_from_slice(&y);
            }
        }
        y
    }
}

/// Sub-pixel shuffle along time or frequency: conv channel `r·C + c` at position `p` becomes
/// channel `c` at position `p·factor + r`.
fn shuffle(x: &[f32], channels: usize, time: usize, freq: usize, along_time: bool) -> Vec<f32> {
    let factor = x.len() / (channels * time * freq);
    let mut out = vec![0f32; x.len()];
    for r in 0..factor {
        for c in 0..channels {
            let src = &x[(r * channels + c) * time * freq..][..time * freq];
            let dst = &mut out[c * time * freq * factor..][..time * freq * factor];
            for t in 0..time {
                for f in 0..freq {
                    let at = if along_time {
                        (t * factor + r) * freq + f
                    } else {
                        t * freq * factor + f * factor + r
                    };
                    dst[at] = src[t * freq + f];
                }
            }
        }
    }
    out
}

struct Decoder {
    dense: DenseBlock,
    up1: ConvBlock,
    up2: ConvBlock,
    /// One 1×1 head (magnitude) or two (phase real, imaginary).
    heads: Vec<(Vec<f32>, f32)>,
}

impl Decoder {
    fn load(w: &Weights, prefix: &str, heads: &[&str]) -> Result<Self> {
        let up1 = Conv::load(w, &format!("{prefix}.up_conv1.0.conv"), 1, (0, 1), (1, 1))?;
        let up2 = Conv::load_along_time(w, &format!("{prefix}.up_conv2.0.conv"))?;
        Ok(Self {
            dense: DenseBlock::load(w, &format!("{prefix}.dense_block"))?,
            up1: ConvBlock::load(w, &format!("{prefix}.up_conv1"), up1)?,
            up2: ConvBlock::load(w, &format!("{prefix}.up_conv2"), up2)?,
            heads: heads
                .iter()
                .map(|h| {
                    Ok((
                        w.get(&format!("{prefix}.{h}.weight"))?,
                        w.get(&format!("{prefix}.{h}.bias"))?[0],
                    ))
                })
                .collect::<Result<_>>()?,
        })
    }

    /// `x` is `[channels][time][freq]`; returns one `[4·time][2·freq]` plane per head.
    fn forward(&self, x: &[f32], time: usize, freq: usize) -> Vec<Vec<f32>> {
        let channels = self.heads[0].0.len();
        let y = self.dense.forward(x, time, freq);
        let mut y = shuffle(
            &self.up1.conv.forward(&y, time, freq),
            channels,
            time,
            freq,
            false,
        );
        self.up1.act.apply(&mut y);
        let freq = 2 * freq;
        let mut y = shuffle(
            &self.up2.conv.forward(&y, time, freq),
            channels,
            time,
            freq,
            true,
        );
        self.up2.act.apply(&mut y);
        let positions = 4 * time * freq;
        self.heads
            .iter()
            .map(|(weight, bias)| {
                let mut out = vec![*bias; positions];
                gemm(
                    false, false, 1, positions, channels, weight, channels, &y, positions, 1.0,
                    &mut out, positions,
                );
                out
            })
            .collect()
    }
}

struct Block {
    time: BiMambaBlock,
    freq: BiMambaBlock,
}

/// Offline RE-USE: the whole recording goes through the model at once.
pub struct Offline {
    encoder_in: ConvBlock,
    encoder_dense: DenseBlock,
    encoder_out: ConvBlock,
    blocks: Vec<Block>,
    mag: Decoder,
    pha: Decoder,
    #[cfg(feature = "gpu")]
    gpu: Option<crate::gpu::Model>,
}

fn make_even(n: usize) -> usize {
    n + n % 2
}

impl Offline {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let w = Weights::load(path)?;
        let mut blocks = Vec::new();
        while w
            .shape(&format!("TSMamba.{}.time_mamba.norm.weight", blocks.len()))
            .is_ok()
        {
            let i = blocks.len();
            blocks.push(Block {
                time: BiMambaBlock::load(&w, &format!("TSMamba.{i}.time_mamba"))?,
                freq: BiMambaBlock::load(&w, &format!("TSMamba.{i}.freq_mamba"))?,
            });
        }
        eyre::ensure!(!blocks.is_empty(), "no TSMamba blocks in the model");
        let enc = "dense_encoder";
        Ok(Self {
            encoder_in: ConvBlock::load(
                &w,
                &format!("{enc}.dense_conv_1"),
                Conv::load(&w, &format!("{enc}.dense_conv_1.0"), 1, (0, 0), (1, 1))?,
            )?,
            encoder_dense: DenseBlock::load(&w, &format!("{enc}.dense_block"))?,
            encoder_out: ConvBlock::load(
                &w,
                &format!("{enc}.dense_conv_2"),
                Conv::load(&w, &format!("{enc}.dense_conv_2.0"), 1, (0, 0), (4, 2))?,
            )?,
            blocks,
            mag: Decoder::load(&w, "mask_decoder", &["final_conv"])?,
            pha: Decoder::load(&w, "phase_decoder", &["phase_conv_r", "phase_conv_i"])?,
            #[cfg(feature = "gpu")]
            gpu: None,
        })
    }

    /// Like [`load`](Self::load), with the TF-Mamba blocks on a GPU. The convolutional encoder
    /// and decoders stay on the CPU, where they run faster.
    #[cfg(feature = "gpu")]
    pub fn load_gpu(path: impl AsRef<Path>, device: crate::GpuDevice) -> Result<Self> {
        let mut model = Self::load(path.as_ref())?;
        let w = Weights::load(path)?;
        let blocks = std::mem::take(&mut model.blocks).len();
        model.gpu = Some(crate::gpu::Model::load(&w, blocks, device)?);
        Ok(model)
    }

    /// `(n_fft, hop)` for `sample_rate`: 320 and 40 at 8 kHz, scaled and rounded up to even
    /// sizes, as NVIDIA's inference does.
    pub fn stft_sizes(sample_rate: u32) -> (usize, usize) {
        let sr = sample_rate as usize;
        (make_even(320 * sr / 8000), make_even(40 * sr / 8000))
    }

    /// The TF-Mamba blocks on `[freq][time][channels]` rows.
    fn mamba(&self, mut rows: Vec<f32>, time: usize, freq: usize, channels: usize) -> Vec<f32> {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &self.gpu {
            return gpu.mamba(&rows, time, freq, channels);
        }
        for block in &self.blocks {
            let t = block.time.forward_batch::<true>(&rows, freq, time);
            rows.iter_mut().zip(&t).for_each(|(r, t)| *r += t);
            rows = transpose_outer(&rows, freq, time, channels);
            let f = block.freq.forward_batch::<true>(&rows, time, freq);
            rows.iter_mut().zip(&f).for_each(|(r, f)| *r += f);
            rows = transpose_outer(&rows, time, freq, channels);
        }
        rows
    }

    /// Model on a compressed magnitude and phase spectrogram (`[freq][time]` each); returns the
    /// enhanced magnitude (before decompression) and phase in the same shape.
    pub fn forward(&self, mag: &[Vec<f32>], pha: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let (bins, frames) = (mag.len(), mag[0].len());
        // Two zero frames and two zero bins at the end, as NVIDIA's model adds.
        let (time, freq) = (frames + 2, bins + 2);
        let mut x = vec![0f32; 2 * time * freq];
        for (ch, spec) in [mag, pha].into_iter().enumerate() {
            for (f, row) in spec.iter().enumerate() {
                for (t, v) in row.iter().enumerate() {
                    x[(ch * time + t) * freq + f] = *v;
                }
            }
        }

        let (h, time, freq) = self.encoder_in.forward(&x, time, freq);
        let h = self.encoder_dense.forward(&h, time, freq);
        let (h, time, freq) = self.encoder_out.forward(&h, time, freq);

        let channels = h.len() / (time * freq);
        // [channels][time][freq] -> [freq][time][channels]
        let rows = transpose_outer(&transpose(&h, channels, time * freq), time, freq, channels);
        let rows = self.mamba(rows, time, freq, channels);
        let rows = transpose_outer(&rows, freq, time, channels);
        let h = transpose(&rows, time * freq, channels);

        let (mag_out, pha_out) = rayon::join(
            || self.mag.forward(&h, time, freq),
            || self.pha.forward(&h, time, freq),
        );
        let width = 2 * freq;
        let crop = |p: &[f32]| -> Vec<Vec<f32>> {
            (0..bins)
                .map(|f| (0..frames).map(|t| p[t * width + f]).collect())
                .collect()
        };
        let phase: Vec<f32> = pha_out[1]
            .iter()
            .zip(&pha_out[0])
            .map(|(i, r)| i.atan2(*r))
            .collect();
        (crop(&mag_out[0]), crop(&phase))
    }

    /// Enhances a whole mono recording at any of [`audio::SAMPLE_RATES`](crate::audio::SAMPLE_RATES).
    /// Memory grows with the length; for long recordings use
    /// [`enhance_chunked`](Self::enhance_chunked).
    pub fn enhance(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        eyre::ensure!(
            SAMPLE_RATES.contains(&sample_rate),
            "unsupported sample rate {sample_rate} Hz, expected one of {SAMPLE_RATES:?}"
        );
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let (n_fft, hop) = Self::stft_sizes(sample_rate);
        let (mut mag, pha) = stft(samples, n_fft, hop, n_fft)?;
        mag.iter_mut().flatten().for_each(|m| *m = m.ln_1p());
        let (mut mag, pha) = self.forward(&mag, &pha);
        // Frames where most bins come out as zero are cleared, as NVIDIA's inference does.
        let (bins, frames) = (mag.len(), mag[0].len());
        let cleared: Vec<bool> = (0..frames)
            .map(|t| mag.iter().filter(|row| row[t] <= 0.0).count() * 2 > bins)
            .collect();
        for row in &mut mag {
            for (m, &clear) in row.iter_mut().zip(&cleared) {
                *m = if clear { 0.0 } else { m.max(0.0).exp_m1() };
            }
        }
        let mut out = istft(&mag, &pha, n_fft, hop, n_fft)?;
        out.resize(samples.len(), 1e-8);
        Ok(out)
    }

    /// Enhances a long recording in overlapping chunks of `chunk_seconds`, starting every
    /// `hop_fraction · chunk_seconds`, blended with a Hann window. Memory stays bounded by the
    /// chunk length; overlap adds work (0.5 processes every sample twice). NVIDIA uses 5 s and
    /// 0.5.
    pub fn enhance_chunked(
        &self,
        samples: &[f32],
        sample_rate: u32,
        chunk_seconds: f32,
        hop_fraction: f32,
    ) -> Result<Vec<f32>> {
        let chunk = (chunk_seconds as f64 * sample_rate as f64) as usize;
        let hop = (hop_fraction as f64 * chunk as f64) as usize;
        eyre::ensure!(
            chunk > 0 && hop > 0,
            "chunk and hop must be at least one sample"
        );
        let window: Vec<f32> = hann_window(chunk).iter().map(|w| w.max(1e-7)).collect();
        let n = samples.len();
        let chunks = if n > chunk {
            (n - chunk).div_ceil(hop) + 1
        } else {
            1
        };
        let mut out = vec![0f32; n];
        let mut weight = vec![0f32; n];
        for i in 0..chunks {
            let start = if i + 1 < chunks {
                i * hop
            } else {
                n.saturating_sub(chunk)
            };
            let end = (start + chunk).min(n);
            let enhanced = self.enhance(&samples[start..end], sample_rate)?;
            for (j, v) in enhanced.iter().enumerate() {
                out[start + j] += v * window[j];
                weight[start + j] += window[j];
            }
        }
        out.iter_mut().zip(&weight).for_each(|(o, w)| *o /= w);
        Ok(out)
    }
}

/// `[a][b][c]` -> `[b][a][c]`.
fn transpose_outer(x: &[f32], a: usize, b: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    out.par_chunks_mut(a * c).enumerate().for_each(|(j, dst)| {
        for i in 0..a {
            dst[i * c..][..c].copy_from_slice(&x[(i * b + j) * c..][..c]);
        }
    });
    out
}
