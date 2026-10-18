//! Real-time RE-USE (nvidia/Real-time_RE-USE), one 20 ms STFT frame per step, look-ahead 0.

use crate::layers::{BiMambaBlock, DenseBlock, LayerNorm, Linear, Mamba, NormAct, StreamConv};
use crate::ops::{gemm, transpose};
use crate::weights::Weights;
use eyre::Result;
use std::path::Path;

/// Causal time Mamba: `norm(output_proj(mamba.step(x) + x))`, one step per frame for every bin.
struct TimeMamba {
    mamba: Mamba,
    output_proj: Linear,
    norm: LayerNorm,
    conv_state: Vec<f32>,
    ssm_state: Vec<f32>,
}

impl TimeMamba {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            mamba: Mamba::load(w, &format!("{prefix}.forward_blocks"))?,
            output_proj: Linear::load(w, &format!("{prefix}.output_proj"), true)?,
            norm: LayerNorm::load(w, &format!("{prefix}.norm"))?,
            conv_state: Vec::new(),
            ssm_state: Vec::new(),
        })
    }

    fn step(&mut self, x: &[f32], rows: usize) -> Vec<f32> {
        let m = &self.mamba;
        if self.ssm_state.len() != rows * m.d_inner * m.d_state {
            self.conv_state = vec![0f32; rows * m.d_inner * m.d_conv];
            self.ssm_state = vec![0f32; rows * m.d_inner * m.d_state];
        }
        let mut y = m.step(x, rows, &mut self.conv_state, &mut self.ssm_state);
        y.iter_mut().zip(x).for_each(|(v, x)| *v += x);
        let mut out = self.output_proj.forward(&y, rows);
        self.norm.apply(&mut out);
        out
    }

    fn reset(&mut self) {
        self.conv_state.clear();
        self.ssm_state.clear();
    }
}

struct Block {
    time: TimeMamba,
    freq: BiMambaBlock,
}

struct Decoder {
    dense: DenseBlock,
    up1: StreamConv,
    up1_act: NormAct,
    up2: StreamConv,
    up2_act: NormAct,
    /// One 1×1 head (magnitude) or two (phase real, imaginary).
    heads: Vec<(Vec<f32>, f32)>,
}

impl Decoder {
    fn load(w: &Weights, prefix: &str, heads: &[&str]) -> Result<Self> {
        Ok(Self {
            dense: DenseBlock::load(w, &format!("{prefix}.dense_block"))?,
            up1: StreamConv::load(w, &format!("{prefix}.up_conv1.0.conv"), 1, 1, 1)?,
            up1_act: NormAct::load(w, &format!("{prefix}.up_conv1.1"), &format!("{prefix}.up_conv1.2"))?,
            up2: StreamConv::load_time_last(w, &format!("{prefix}.up_conv2.0.conv"))?,
            up2_act: NormAct::load(w, &format!("{prefix}.up_conv2.1"), &format!("{prefix}.up_conv2.2"))?,
            heads: heads
                .iter()
                .map(|h| Ok((w.get(&format!("{prefix}.{h}.weight"))?, w.get(&format!("{prefix}.{h}.bias"))?[0])))
                .collect::<Result<_>>()?,
        })
    }

    fn reset(&mut self) {
        self.dense.reset();
        self.up2.reset();
    }

    /// `x` is `[channels][freq]`; returns one `[2·freq]` row per head.
    fn step(&mut self, x: &[f32]) -> Vec<Vec<f32>> {
        let y = self.dense.step(x);
        let up = self.up1.step(&y);
        // Sub-pixel shuffle: conv channel r·C + c at bin w becomes channel c at bin 2w + r.
        let channels = self.heads[0].0.len();
        let freq = y.len() / channels;
        let mut shuffled = vec![0f32; up.len()];
        for r in 0..2 {
            for c in 0..channels {
                for w in 0..freq {
                    shuffled[c * 2 * freq + 2 * w + r] = up[(r * channels + c) * freq + w];
                }
            }
        }
        self.up1_act.apply(&mut shuffled);
        let mut z = self.up2.step(&shuffled);
        self.up2_act.apply(&mut z);
        let positions = 2 * freq;
        self.heads
            .iter()
            .map(|(weight, bias)| {
                let mut out = vec![*bias; positions];
                gemm(false, false, 1, positions, channels, weight, channels, &z, positions, 1.0, &mut out, positions);
                out
            })
            .collect()
    }
}

/// Streaming real-time RE-USE on the CPU.
pub struct Streaming {
    encoder_in: StreamConv,
    encoder_in_act: NormAct,
    encoder_dense: DenseBlock,
    encoder_out: StreamConv,
    encoder_out_act: NormAct,
    blocks: Vec<Block>,
    mag: Decoder,
    pha: Decoder,
}

impl Streaming {
    /// Loads the first `layers` TF-Mamba blocks (3–12) and the decoder pair trained to exit there.
    pub fn load(path: impl AsRef<Path>, layers: usize) -> Result<Self> {
        let w = Weights::load(path)?;
        eyre::ensure!((3..=12).contains(&layers), "layers must be 3..=12, got {layers}");
        let blocks = (0..layers)
            .map(|i| {
                Ok(Block {
                    time: TimeMamba::load(&w, &format!("TSMamba.{i}.time_mamba"))?,
                    freq: BiMambaBlock::load(&w, &format!("TSMamba.{i}.freq_mamba"))?,
                })
            })
            .collect::<Result<_>>()?;
        let exit = layers - 1;
        Ok(Self {
            encoder_in: StreamConv::load(&w, "dense_encoder.dense_conv_1_1.0", 1, 1, 1)?,
            encoder_in_act: NormAct::load(&w, "dense_encoder.dense_conv_1_1.1", "dense_encoder.dense_conv_1_1.2")?,
            encoder_dense: DenseBlock::load(&w, "dense_encoder.dense_block")?,
            encoder_out: StreamConv::load(&w, "dense_encoder.dense_conv_2.0", 1, 0, 2)?,
            encoder_out_act: NormAct::load(&w, "dense_encoder.dense_conv_2.1", "dense_encoder.dense_conv_2.2")?,
            blocks,
            mag: Decoder::load(&w, &format!("mask_decoder_list.{exit}"), &["final_conv"])?,
            pha: Decoder::load(&w, &format!("phase_decoder_list.{exit}"), &["phase_conv_r", "phase_conv_i"])?,
        })
    }

    pub fn reset(&mut self) {
        self.encoder_in.reset();
        self.encoder_dense.reset();
        for b in &mut self.blocks {
            b.time.reset();
        }
        self.mag.reset();
        self.pha.reset();
    }

    /// One frame: log1p-compressed magnitude and phase (`bins` each) in, enhanced ones out.
    pub fn step(&mut self, mag: &[f32], pha: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let bins = mag.len();
        let freq_in = bins + 2;
        let mut x = vec![0f32; 2 * freq_in];
        x[..bins].copy_from_slice(mag);
        x[freq_in..freq_in + bins].copy_from_slice(pha);

        let mut h = self.encoder_in.step(&x);
        self.encoder_in_act.apply(&mut h);
        let h = self.encoder_dense.step(&h);
        let mut h = self.encoder_out.step(&h);
        self.encoder_out_act.apply(&mut h);

        let channels = self.encoder_out.cout;
        let freq = h.len() / channels;
        let mut rows = transpose(&h, channels, freq);
        for block in &mut self.blocks {
            let t = block.time.step(&rows, freq);
            rows.iter_mut().zip(&t).for_each(|(r, t)| *r += t);
            let f = block.freq.forward(&rows, freq);
            rows.iter_mut().zip(&f).for_each(|(r, f)| *r += f);
        }
        let h = transpose(&rows, freq, channels);

        let (mag_dec, pha_dec) = (&mut self.mag, &mut self.pha);
        let (mut mag, pha) = rayon::join(|| mag_dec.step(&h), || pha_dec.step(&h));
        let mag = mag.swap_remove(0);
        let phase = pha[0].iter().zip(&pha[1]).map(|(r, i)| i.atan2(*r)).collect::<Vec<_>>();
        (mag[..bins].to_vec(), phase[..bins].to_vec())
    }
}
