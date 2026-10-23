//! RE-USE building blocks. Conv activations are channel-major `[channels][freq]`;
//! Mamba activations are position-major `[positions][channels]`.

use crate::ops::{
    add_channel_bias, fast_exp, fast_exp_fma, gemm, layer_norm_prelu_channels, layer_norm_rows,
    silu, softplus, softplus_fma,
};
use crate::weights::Weights;
use eyre::Result;
use rayon::prelude::*;
use std::collections::VecDeque;

/// Conv2d over (time, freq) for one new time frame. Time is causal: the last `(taps - 1) * dilation`
/// input frames are cached. Frequency uses `freq_pad` zeros on both sides and `stride`.
pub struct StreamConv {
    weight: Vec<f32>,
    bias: Vec<f32>,
    cin: usize,
    pub cout: usize,
    taps: usize,
    dilation: usize,
    kf: usize,
    freq_pad: usize,
    stride: usize,
    cache: VecDeque<Vec<f32>>,
}

impl StreamConv {
    /// `prefix.weight` is `[cout][cin][taps][kf]`.
    pub fn load(
        w: &Weights,
        prefix: &str,
        dilation: usize,
        freq_pad: usize,
        stride: usize,
    ) -> Result<Self> {
        let shape = w.shape(&format!("{prefix}.weight"))?.to_vec();
        Ok(Self {
            weight: w.get(&format!("{prefix}.weight"))?,
            bias: w.get(&format!("{prefix}.bias"))?,
            cout: shape[0],
            cin: shape[1],
            taps: shape[2],
            kf: shape[3],
            dilation,
            freq_pad,
            stride,
            cache: VecDeque::new(),
        })
    }

    /// Same as `load` for a kernel stored as `[cout][cin][1][taps]` (time on the last axis).
    pub fn load_time_last(w: &Weights, prefix: &str) -> Result<Self> {
        let mut conv = Self::load(w, prefix, 1, 0, 1)?;
        conv.taps = conv.kf;
        conv.kf = 1;
        Ok(conv)
    }

    pub fn reset(&mut self) {
        self.cache.clear();
    }

    /// Adds `x` to the time history without computing an output. On an empty history, `zeros`
    /// zero frames come first.
    pub fn push(&mut self, x: &[f32], zeros: usize) {
        if self.cache.is_empty() {
            for _ in 0..zeros {
                self.cache.push_back(vec![0f32; x.len()]);
            }
        }
        self.cache.push_back(x.to_vec());
    }

    /// `x` is `[cin][freq]`; returns `[cout][freq_out]`.
    pub fn step(&mut self, x: &[f32]) -> Vec<f32> {
        let freq = x.len() / self.cin;
        let history = (self.taps - 1) * self.dilation;
        while self.cache.len() < history {
            self.cache.push_back(vec![0f32; x.len()]);
        }
        let freq_out = (freq + 2 * self.freq_pad - self.kf) / self.stride + 1;
        let rows = self.cin * self.taps * self.kf;
        let mut cols = vec![0f32; rows * freq_out];
        for tap in 0..self.taps {
            let at = tap * self.dilation;
            let frame: &[f32] = if at < history { &self.cache[at] } else { x };
            for ci in 0..self.cin {
                let src = &frame[ci * freq..(ci + 1) * freq];
                for k in 0..self.kf {
                    let row =
                        &mut cols[((ci * self.taps + tap) * self.kf + k) * freq_out..][..freq_out];
                    for (fo, out) in row.iter_mut().enumerate() {
                        let fi = (fo * self.stride + k) as isize - self.freq_pad as isize;
                        if fi >= 0 && (fi as usize) < freq {
                            *out = src[fi as usize];
                        }
                    }
                }
            }
        }
        let mut out = vec![0f32; self.cout * freq_out];
        gemm(
            false,
            false,
            self.cout,
            freq_out,
            rows,
            &self.weight,
            rows,
            &cols,
            freq_out,
            0.0,
            &mut out,
            freq_out,
        );
        add_channel_bias(&mut out, &self.bias);
        if history > 0 {
            self.cache.pop_front();
            self.cache.push_back(x.to_vec());
        }
        out
    }
}

/// ChannelLayerNorm followed by PReLU.
pub struct NormAct {
    weight: Vec<f32>,
    bias: Vec<f32>,
    prelu: Vec<f32>,
}

impl NormAct {
    pub fn load(w: &Weights, norm: &str, prelu: &str) -> Result<Self> {
        Ok(Self {
            weight: w.get(&format!("{norm}.norm.weight"))?,
            bias: w.get(&format!("{norm}.norm.bias"))?,
            prelu: w.get(&format!("{prelu}.weight"))?,
        })
    }

    pub fn apply(&self, x: &mut [f32]) {
        layer_norm_prelu_channels(x, self.weight.len(), &self.weight, &self.bias, &self.prelu);
    }
}

/// Causal DenseBlock: 4 dilated 3×3 convs, each seeing all earlier outputs and the input.
pub struct DenseBlock {
    layers: Vec<(StreamConv, NormAct)>,
    channels: usize,
}

impl DenseBlock {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..4 {
            let p = format!("{prefix}.dense_block.{i}");
            layers.push((
                StreamConv::load(w, &format!("{p}.1"), 1 << i, 1, 1)?,
                NormAct::load(w, &format!("{p}.2"), &format!("{p}.3"))?,
            ));
        }
        let channels = w.shape(&format!("{prefix}.dense_block.0.1.weight"))?[0];
        Ok(Self { layers, channels })
    }

    pub fn reset(&mut self) {
        self.layers.iter_mut().for_each(|(c, _)| c.reset());
    }

    /// `x` is `[channels][freq]`. Layer `i` reads `[y_{i-1}, …, y_0, x]`, which is kept as one
    /// contiguous tail of `buf`.
    pub fn step(&mut self, x: &[f32]) -> Vec<f32> {
        let size = x.len();
        let depth = self.layers.len();
        let mut buf = vec![0f32; (depth + 1) * size];
        buf[depth * size..].copy_from_slice(x);
        let mut y = Vec::new();
        for (i, (conv, act)) in self.layers.iter_mut().enumerate() {
            y = conv.step(&buf[(depth - i) * size..]);
            act.apply(&mut y);
            if i + 1 < depth {
                buf[(depth - 1 - i) * size..(depth - i) * size].copy_from_slice(&y);
            }
        }
        debug_assert_eq!(y.len(), self.channels * (size / self.channels));
        y
    }
}

/// Linear layer `y = x·Wᵀ + b` on position-major rows.
pub struct Linear {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    pub out: usize,
    pub inp: usize,
}

impl Linear {
    pub fn load(w: &Weights, prefix: &str, bias: bool) -> Result<Self> {
        let shape = w.shape(&format!("{prefix}.weight"))?.to_vec();
        Ok(Self {
            weight: w.get(&format!("{prefix}.weight"))?,
            bias: if bias {
                Some(w.get(&format!("{prefix}.bias"))?)
            } else {
                None
            },
            out: shape[0],
            inp: shape[1],
        })
    }

    /// `x` holds `rows` rows of `inp` values, each starting `ld` apart.
    pub fn forward_strided(&self, x: &[f32], rows: usize, ld: usize) -> Vec<f32> {
        let mut y = vec![0f32; rows * self.out];
        gemm(
            false,
            true,
            rows,
            self.out,
            self.inp,
            x,
            ld,
            &self.weight,
            self.inp,
            0.0,
            &mut y,
            self.out,
        );
        if let Some(b) = &self.bias {
            for row in y.chunks_exact_mut(self.out) {
                row.iter_mut().zip(b).for_each(|(v, b)| *v += b);
            }
        }
        y
    }

    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        self.forward_strided(x, rows, self.inp)
    }
}

/// SSM state size; RE-USE uses 16 for every Mamba.
const STATE: usize = 16;

fn state16(x: &[f32]) -> &[f32; STATE] {
    x[..STATE].try_into().unwrap()
}

/// One selective-scan update of a channel: `s = exp(Δ·A)·s + Δx·B`, returns `C·s`.
/// The exps, the update and the sum are separate loops so each one vectorizes.
#[inline(always)]
fn scan16(
    s: &mut [f32; STATE],
    a: &[f32; STATE],
    delta: f32,
    dx: f32,
    b: &[f32; STATE],
    c: &[f32; STATE],
) -> f32 {
    let mut decay = [0f32; STATE];
    for n in 0..STATE {
        decay[n] = fast_exp(delta * a[n]);
    }
    let mut out = [0f32; STATE];
    for n in 0..STATE {
        s[n] = decay[n] * s[n] + dx * b[n];
        out[n] = s[n] * c[n];
    }
    let mut acc = [0f32; 4];
    for n in 0..STATE {
        acc[n % 4] += out[n];
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

/// Mamba-1 (mamba_ssm.Mamba) with its parameter names.
pub struct Mamba {
    in_proj: Linear,
    conv_w: Vec<f32>,
    /// `conv_w` as `[d_conv][d_inner]`, so a tap runs over contiguous channels.
    conv_taps: Vec<f32>,
    conv_b: Vec<f32>,
    x_proj: Linear,
    dt_proj: Linear,
    out_proj: Linear,
    a: Vec<f32>,
    /// `a` as `[d_state][d_inner]`, for the scan that runs across channels.
    a_t: Vec<f32>,
    d: Vec<f32>,
    pub d_inner: usize,
    pub d_state: usize,
    pub d_conv: usize,
    dt_rank: usize,
}

impl Mamba {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self> {
        let a_log = w.get(&format!("{prefix}.A_log"))?;
        let a_shape = w.shape(&format!("{prefix}.A_log"))?.to_vec();
        let conv_shape = w.shape(&format!("{prefix}.conv1d.weight"))?.to_vec();
        let dt_proj = Linear::load(w, &format!("{prefix}.dt_proj"), true)?;
        let conv_w = w.get(&format!("{prefix}.conv1d.weight"))?;
        let (di, dc) = (conv_shape[0], conv_shape[2]);
        let conv_taps = (0..dc * di)
            .map(|i| conv_w[(i % di) * dc + i / di])
            .collect();
        eyre::ensure!(
            a_shape[1] == STATE,
            "{prefix}: d_state {} is not {STATE}",
            a_shape[1]
        );
        Ok(Self {
            in_proj: Linear::load(w, &format!("{prefix}.in_proj"), false)?,
            conv_w,
            conv_taps,
            conv_b: w.get(&format!("{prefix}.conv1d.bias"))?,
            x_proj: Linear::load(w, &format!("{prefix}.x_proj"), false)?,
            out_proj: Linear::load(w, &format!("{prefix}.out_proj"), false)?,
            a_t: (0..a_log.len())
                .map(|i| -a_log[(i % di) * STATE + i / di].exp())
                .collect(),
            a: a_log.iter().map(|v| -v.exp()).collect(),
            d: w.get(&format!("{prefix}.D"))?,
            d_inner: a_shape[0],
            d_state: a_shape[1],
            d_conv: conv_shape[2],
            dt_rank: dt_proj.inp,
            dt_proj,
        })
    }

    /// x_proj and dt_proj on the conv output `xc` (`rows × d_inner`): returns (dt, dbl) where
    /// `dbl` rows are `[dt_raw | B | C]`.
    fn ssm_inputs<const FAST: bool>(&self, xc: &[f32], rows: usize) -> (Vec<f32>, Vec<f32>) {
        let dbl = self.x_proj.forward(xc, rows);
        let mut dt = self.dt_proj.forward_strided(&dbl, rows, self.x_proj.out);
        dt.par_chunks_mut(4096).for_each(|c| {
            for v in c {
                *v = if FAST { softplus_fma(*v) } else { softplus(*v) };
            }
        });
        (dt, dbl)
    }

    /// `(y + D·x) · silu(z)`, then out_proj.
    fn finish(&self, mut y: Vec<f32>, xc: &[f32], xz: &[f32], rows: usize) -> Vec<f32> {
        let di = self.d_inner;
        y.par_chunks_mut(di)
            .zip(xc.par_chunks(di))
            .zip(xz.par_chunks(2 * di))
            .for_each(|((y, xc), xz)| {
                for (((y, x), z), d) in y.iter_mut().zip(xc).zip(&xz[di..]).zip(&self.d) {
                    *y = (*y + d * x) * silu(*z);
                }
            });
        self.out_proj.forward(&y, rows)
    }

    /// One state update and output, for each of `rows` independent sequences.
    /// `x` is `rows × d_model`; `conv_state` is `rows × d_inner × d_conv`; `ssm_state` is
    /// `rows × d_inner × d_state`.
    pub fn step(
        &self,
        x: &[f32],
        rows: usize,
        conv_state: &mut [f32],
        ssm_state: &mut [f32],
    ) -> Vec<f32> {
        let (di, ds, dc) = (self.d_inner, self.d_state, self.d_conv);
        let xz = self.in_proj.forward(x, rows);
        let mut xc = vec![0f32; rows * di];
        for r in 0..rows {
            for c in 0..di {
                let state = &mut conv_state[(r * di + c) * dc..][..dc];
                state.copy_within(1.., 0);
                state[dc - 1] = xz[r * 2 * di + c];
                let w = &self.conv_w[c * dc..][..dc];
                let acc: f32 = state.iter().zip(w).map(|(s, w)| s * w).sum();
                xc[r * di + c] = silu(acc + self.conv_b[c]);
            }
        }
        let (dt, dbl) = self.ssm_inputs::<false>(&xc, rows);
        let nd = self.x_proj.out;
        let mut y = vec![0f32; rows * di];
        y.par_chunks_mut(di)
            .zip(ssm_state.par_chunks_mut(di * ds))
            .enumerate()
            .for_each(|(r, (y, state))| {
                let b = state16(&dbl[r * nd + self.dt_rank..]);
                let cm = state16(&dbl[r * nd + self.dt_rank + ds..]);
                for c in 0..di {
                    let delta = dt[r * di + c];
                    let s: &mut [f32; STATE] = (&mut state[c * ds..][..ds]).try_into().unwrap();
                    y[c] = scan16(
                        s,
                        state16(&self.a[c * ds..]),
                        delta,
                        delta * xc[r * di + c],
                        b,
                        cm,
                    );
                }
            });
        self.finish(y, &xc, &xz, rows)
    }

    /// Selective scan of channels `c0..c0 + 32` from zero state over the rows of `y`
    /// (`len × d_inner`), which start at row `start`. Each vector lane is one channel, so `C·s`
    /// needs no horizontal sum.
    fn scan_channels(
        &self,
        dt: &[f32],
        dbl: &[f32],
        xc: &[f32],
        start: usize,
        c0: usize,
        y: &mut [f32],
    ) {
        const W: usize = 32;
        let (di, nd) = (self.d_inner, self.x_proj.out);
        let lanes = |v: &[f32], at: usize| -> [f32; W] { v[at..][..W].try_into().unwrap() };
        let a: [[f32; W]; STATE] = std::array::from_fn(|n| lanes(&self.a_t, n * di + c0));
        let mut s = [[0f32; W]; STATE];
        for (l, out) in y.chunks_exact_mut(di).enumerate() {
            let row = start + l;
            let b = state16(&dbl[row * nd + self.dt_rank..]);
            let cm = state16(&dbl[row * nd + self.dt_rank + STATE..]);
            let delta = lanes(dt, row * di + c0);
            let x = lanes(xc, row * di + c0);
            let dx: [f32; W] = std::array::from_fn(|j| delta[j] * x[j]);
            let mut acc = [0f32; W];
            for n in 0..STATE {
                for j in 0..W {
                    let decay = fast_exp_fma(delta[j] * a[n][j]);
                    s[n][j] = decay.mul_add(s[n][j], dx[j] * b[n]);
                    acc[j] = s[n][j].mul_add(cm[n], acc[j]);
                }
            }
            out[c0..][..W].copy_from_slice(&acc);
        }
    }

    /// Full causal passes over `batch` independent sequences, each starting from zero state.
    /// `x` is `batch × len × d_model`.
    ///
    /// With `FAST`, the scan runs across channels with fused multiply-adds, which rounds
    /// slightly differently.
    pub fn sequences<const FAST: bool>(&self, x: &[f32], batch: usize, len: usize) -> Vec<f32> {
        let (di, ds, dc) = (self.d_inner, self.d_state, self.d_conv);
        let rows = batch * len;
        let xz = self.in_proj.forward(x, rows);
        let mut xc = vec![0f32; rows * di];
        xc.par_chunks_mut(len * di)
            .zip(xz.par_chunks(len * 2 * di))
            .for_each(|(xc, xz)| {
                for (l, out) in xc.chunks_exact_mut(di).enumerate() {
                    out.copy_from_slice(&self.conv_b);
                    for k in (dc - 1).saturating_sub(l)..dc {
                        let src = &xz[(l + k + 1 - dc) * 2 * di..][..di];
                        let taps = &self.conv_taps[k * di..][..di];
                        for ((o, w), x) in out.iter_mut().zip(taps).zip(src) {
                            *o += w * x;
                        }
                    }
                    out.iter_mut().for_each(|v| *v = silu(*v));
                }
            });
        let (dt, dbl) = self.ssm_inputs::<FAST>(&xc, rows);
        let nd = self.x_proj.out;
        // Channels are independent: each block scans its channels over one whole sequence.
        const BLOCK: usize = 32;
        if FAST && di % BLOCK == 0 {
            let mut y = vec![0f32; rows * di];
            y.par_chunks_mut(len * di).enumerate().for_each(|(seq, y)| {
                for c0 in (0..di).step_by(BLOCK) {
                    self.scan_channels(&dt, &dbl, &xc, seq * len, c0, y);
                }
            });
            return self.finish(y, &xc, &xz, rows);
        }
        let per_seq = di.div_ceil(BLOCK);
        let blocks: Vec<Vec<f32>> = (0..batch * per_seq)
            .into_par_iter()
            .map(|task| {
                let (seq, blk) = (task / per_seq, task % per_seq);
                let channels = blk * BLOCK..((blk + 1) * BLOCK).min(di);
                let width = channels.len();
                let mut state = vec![[0f32; STATE]; width];
                let mut out = vec![0f32; len * width];
                for l in 0..len {
                    let row = seq * len + l;
                    let b = state16(&dbl[row * nd + self.dt_rank..]);
                    let cm = state16(&dbl[row * nd + self.dt_rank + ds..]);
                    for (j, c) in channels.clone().enumerate() {
                        let delta = dt[row * di + c];
                        out[l * width + j] = scan16(
                            &mut state[j],
                            state16(&self.a[c * ds..]),
                            delta,
                            delta * xc[row * di + c],
                            b,
                            cm,
                        );
                    }
                }
                out
            })
            .collect();
        let mut y = vec![0f32; rows * di];
        for (task, out) in blocks.iter().enumerate() {
            let (seq, blk) = (task / per_seq, task % per_seq);
            let width = out.len() / len;
            for l in 0..len {
                y[(seq * len + l) * di + blk * BLOCK..][..width]
                    .copy_from_slice(&out[l * width..][..width]);
            }
        }
        self.finish(y, &xc, &xz, rows)
    }
}

/// LayerNorm over the last dimension.
pub struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl LayerNorm {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: w.get(&format!("{prefix}.weight"))?,
            bias: w.get(&format!("{prefix}.bias"))?,
        })
    }

    pub fn apply(&self, x: &mut [f32]) {
        let dim = self.weight.len();
        x.par_chunks_mut(dim * 1024)
            .for_each(|rows| layer_norm_rows(rows, dim, &self.weight, &self.bias));
    }
}

/// Bidirectional MambaBlock: `norm(output_proj([fw(x) + x, flip(bw(flip x)) + x]))`.
pub struct BiMambaBlock {
    forward: Mamba,
    backward: Mamba,
    output_proj: Linear,
    norm: LayerNorm,
}

impl BiMambaBlock {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            forward: Mamba::load(w, &format!("{prefix}.forward_blocks"))?,
            backward: Mamba::load(w, &format!("{prefix}.backward_blocks"))?,
            output_proj: Linear::load(w, &format!("{prefix}.output_proj"), true)?,
            norm: LayerNorm::load(w, &format!("{prefix}.norm"))?,
        })
    }

    /// `x` is `len × d_model`.
    pub fn forward(&self, x: &[f32], len: usize) -> Vec<f32> {
        self.forward_batch::<false>(x, 1, len)
    }

    /// [`forward`](Self::forward) for `batch` independent sequences, `x` is
    /// `batch × len × d_model`.
    pub fn forward_batch<const FAST: bool>(&self, x: &[f32], batch: usize, len: usize) -> Vec<f32> {
        let rows = batch * len;
        let dm = x.len() / rows;
        let mut reversed = vec![0f32; x.len()];
        reversed
            .par_chunks_mut(len * dm)
            .zip(x.par_chunks(len * dm))
            .for_each(|(dst, src)| {
                for (d, s) in dst.chunks_exact_mut(dm).zip(src.chunks_exact(dm).rev()) {
                    d.copy_from_slice(s);
                }
            });
        let (fw, bw) = rayon::join(
            || self.forward.sequences::<FAST>(x, batch, len),
            || self.backward.sequences::<FAST>(&reversed, batch, len),
        );
        let mut cat = vec![0f32; rows * 2 * dm];
        cat.par_chunks_mut(len * 2 * dm)
            .enumerate()
            .for_each(|(s, cat)| {
                for l in 0..len {
                    let r = s * len + l;
                    let row = &mut cat[l * 2 * dm..][..2 * dm];
                    let xin = &x[r * dm..][..dm];
                    let back = &bw[(s * len + len - 1 - l) * dm..][..dm];
                    for c in 0..dm {
                        row[c] = fw[r * dm + c] + xin[c];
                        row[dm + c] = back[c] + xin[c];
                    }
                }
            });
        let mut out = self.output_proj.forward(&cat, rows);
        self.norm.apply(&mut out);
        out
    }
}
