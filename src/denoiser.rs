//! Samples in, enhanced samples out: streaming STFT, [`Streaming`], inverse STFT.

use crate::audio::{hann_window, stft_sizes};
use crate::Streaming;
use eyre::{eyre, Result};
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::path::Path;
use std::sync::Arc;

/// Settings for [`Denoiser::with_config`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// Mamba layers, 3–12. Fewer is faster.
    pub layers: usize,
    /// Mono sample rate, one of [`audio::SAMPLE_RATES`](crate::audio::SAMPLE_RATES).
    pub sample_rate: u32,
    /// Future frames the model waits for, 0–2. Each one adds a frame (20 ms) of latency.
    pub look_ahead: usize,
}

impl Default for Config {
    /// 8 layers, 16 kHz, no look-ahead.
    fn default() -> Self {
        Self {
            layers: 8,
            sample_rate: 16000,
            look_ahead: 0,
        }
    }
}

/// Streaming speech enhancement for mono audio.
///
/// Feed any number of samples to [`process`](Self::process); it returns the enhanced samples that
/// are complete so far, in blocks of [`frame`](Self::frame) samples. Output sample `i` is the
/// enhanced input sample `i`, delayed by [`latency`](Self::latency) samples. Call
/// [`flush`](Self::flush) at the end of a stream to get the remaining samples.
pub struct Denoiser {
    model: Streaming,
    config: Config,
    n_fft: usize,
    hop: usize,
    window: Vec<f32>,
    /// Overlap-add normalization for each output position: `Σ w²(j + k·hop)`.
    envelope: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    ifft: Arc<dyn ComplexToReal<f32>>,
    /// Input for the next frame; the first `filled` samples are set.
    history: Vec<f32>,
    filled: usize,
    /// Overlap-add accumulator, starting at the first output position not yet emitted.
    overlap: Vec<f32>,
    /// Output positions still to drop: the first frame is centred on sample 0.
    skip: usize,
    /// Input samples not yet returned.
    owed: usize,
    time: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    mag: Vec<f32>,
    pha: Vec<f32>,
}

impl Denoiser {
    /// Loads the model for 16 kHz audio with `layers` Mamba layers (3–12) and no look-ahead.
    ///
    /// Fewer layers are faster, because each frame passes through every layer. The model was
    /// trained to stop after any of these layer counts, so every choice gives usable output.
    pub fn load(path: impl AsRef<Path>, layers: usize) -> Result<Self> {
        Self::with_config(
            path,
            Config {
                layers,
                ..Config::default()
            },
        )
    }

    /// Loads the model with any sample rate and look-ahead.
    pub fn with_config(path: impl AsRef<Path>, config: Config) -> Result<Self> {
        let (n_fft, hop) = stft_sizes(config.sample_rate)?;
        let model = Streaming::load(path, config.layers, config.look_ahead)?;
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);
        let ifft = planner.plan_fft_inverse(n_fft);
        let window = hann_window(n_fft);
        let envelope = (0..hop)
            .map(|j| (j..n_fft).step_by(hop).map(|i| window[i] * window[i]).sum())
            .collect();
        let scratch_len = fft.get_scratch_len().max(ifft.get_scratch_len());
        let bins = n_fft / 2 + 1;
        Ok(Self {
            model,
            config,
            n_fft,
            hop,
            window,
            envelope,
            history: vec![0f32; n_fft],
            filled: n_fft / 2,
            overlap: vec![0f32; n_fft],
            skip: n_fft / 2,
            owed: 0,
            time: vec![0f32; n_fft],
            spectrum: vec![Complex::default(); bins],
            scratch: vec![Complex::default(); scratch_len],
            mag: vec![0f32; bins],
            pha: vec![0f32; bins],
            fft,
            ifft,
        })
    }

    pub fn config(&self) -> Config {
        self.config
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Samples consumed and produced per model step (20 ms).
    pub fn frame(&self) -> usize {
        self.hop
    }

    /// Delay between an input sample and its enhanced output, in samples: one frame, plus one
    /// more per look-ahead frame.
    pub fn latency(&self) -> usize {
        self.n_fft - self.hop + self.config.look_ahead * self.hop
    }

    /// Clears all stream state; the next sample starts a new stream.
    pub fn reset(&mut self) {
        self.model.reset();
        self.history.fill(0.0);
        self.filled = self.n_fft / 2;
        self.overlap.fill(0.0);
        self.skip = self.n_fft / 2;
        self.owed = 0;
    }

    /// Enhances `input` and returns every output sample that is complete.
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(input.len() + self.hop);
        for &sample in input {
            self.history[self.filled] = sample;
            self.filled += 1;
            if self.filled == self.n_fft {
                self.step(&mut out)?;
            }
        }
        self.owed += input.len();
        self.owed -= out.len();
        Ok(out)
    }

    /// Pushes silence through the model so every sample given so far comes out, then resets.
    pub fn flush(&mut self) -> Result<Vec<f32>> {
        let owed = self.owed;
        let silence = vec![0f32; self.hop];
        let mut out = Vec::with_capacity(owed + self.hop);
        while out.len() < owed {
            out.extend(self.process(&silence)?);
        }
        out.truncate(owed);
        self.reset();
        Ok(out)
    }

    /// One STFT frame from `history`: analyse, enhance, overlap-add, emit one finished block.
    fn step(&mut self, out: &mut Vec<f32>) -> Result<()> {
        for ((t, x), w) in self.time.iter_mut().zip(&self.history).zip(&self.window) {
            *t = x * w;
        }
        self.history.copy_within(self.hop.., 0);
        self.filled = self.n_fft - self.hop;
        self.fft
            .process_with_scratch(&mut self.time, &mut self.spectrum, &mut self.scratch)
            .map_err(|e| eyre!("FFT failed: {e}"))?;
        for ((c, m), p) in self.spectrum.iter().zip(&mut self.mag).zip(&mut self.pha) {
            *m = c.norm().ln_1p();
            *p = c.im.atan2(c.re);
        }

        let Some((mut mag, pha)) = self.model.step(&self.mag, &self.pha) else {
            return Ok(());
        };
        // Frames where most bins come out as zero are cleared, as NVIDIA's inference does.
        if mag.iter().filter(|m| **m <= 0.0).count() * 2 > mag.len() {
            mag.fill(0.0);
        }
        for ((c, m), p) in self.spectrum.iter_mut().zip(&mag).zip(&pha) {
            *c = Complex::from_polar(m.max(0.0).exp_m1(), *p);
        }
        let last = self.spectrum.len() - 1;
        self.spectrum[0].im = 0.0;
        self.spectrum[last].im = 0.0;
        self.ifft
            .process_with_scratch(&mut self.spectrum, &mut self.time, &mut self.scratch)
            .map_err(|e| eyre!("IFFT failed: {e}"))?;

        let norm = 1.0 / self.n_fft as f32;
        for ((o, t), w) in self.overlap.iter_mut().zip(&self.time).zip(&self.window) {
            *o += t * norm * w;
        }
        let hop = self.hop;
        let skip = self.skip.min(hop);
        out.extend(
            self.overlap[skip..hop]
                .iter()
                .zip(&self.envelope[skip..])
                .map(|(o, e)| o / e),
        );
        self.skip -= skip;
        self.overlap.copy_within(hop.., 0);
        let n_fft = self.n_fft;
        self.overlap[n_fft - hop..].fill(0.0);
        Ok(())
    }
}
