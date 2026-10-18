//! Samples in, enhanced samples out: streaming STFT, [`Streaming`], inverse STFT.

use crate::audio::hann_window;
use crate::Streaming;
use eyre::{eyre, Result};
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::path::Path;
use std::sync::Arc;

const N_FFT: usize = 640;
const HOP: usize = 320;

/// Streaming speech enhancement for 16 kHz mono audio.
///
/// Feed any number of samples to [`process`](Self::process); it returns the enhanced samples that
/// are complete so far, always a multiple of [`FRAME`](Self::FRAME). Output sample `i` is the
/// enhanced input sample `i`, delayed by [`LATENCY`](Self::LATENCY) samples. Call
/// [`flush`](Self::flush) at the end of a stream to get the remaining samples.
pub struct Denoiser {
    model: Streaming,
    window: Vec<f32>,
    /// `w²(j) + w²(j + HOP)`: overlap-add normalization for each output position.
    envelope: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    ifft: Arc<dyn ComplexToReal<f32>>,
    /// Last `N_FFT` input samples; the newest `pending` of them are not yet analysed.
    history: Vec<f32>,
    pending: usize,
    /// Overlap-add accumulator for the current and next output block.
    overlap: Vec<f32>,
    /// The first output block covers the zero padding before the stream and is dropped.
    started: bool,
    time: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    mag: Vec<f32>,
    pha: Vec<f32>,
}

impl Denoiser {
    /// Sample rate the denoiser runs at (mono).
    pub const SAMPLE_RATE: u32 = 16000;
    /// Samples consumed and produced per model step (20 ms).
    pub const FRAME: usize = HOP;
    /// Delay between an input sample and its enhanced output, in samples (20 ms).
    pub const LATENCY: usize = HOP;

    /// Loads the model with `layers` Mamba layers (3–12; fewer is faster).
    pub fn load(path: impl AsRef<Path>, layers: usize) -> Result<Self> {
        Ok(Self::new(Streaming::load(path, layers)?))
    }

    pub fn new(model: Streaming) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let ifft = planner.plan_fft_inverse(N_FFT);
        let window = hann_window(N_FFT);
        let envelope = (0..HOP)
            .map(|j| window[j] * window[j] + window[j + HOP] * window[j + HOP])
            .collect();
        let scratch_len = fft.get_scratch_len().max(ifft.get_scratch_len());
        let bins = N_FFT / 2 + 1;
        Self {
            model,
            window,
            envelope,
            history: vec![0f32; N_FFT],
            pending: HOP,
            overlap: vec![0f32; N_FFT],
            started: false,
            time: vec![0f32; N_FFT],
            spectrum: vec![Complex::default(); bins],
            scratch: vec![Complex::default(); scratch_len],
            mag: vec![0f32; bins],
            pha: vec![0f32; bins],
            fft,
            ifft,
        }
    }

    /// Clears all stream state; the next sample starts a new stream.
    pub fn reset(&mut self) {
        self.model.reset();
        self.history.fill(0.0);
        self.pending = HOP;
        self.overlap.fill(0.0);
        self.started = false;
    }

    /// Enhances `input` and returns every output sample that is complete.
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(input.len() + HOP);
        for &sample in input {
            self.history.copy_within(1.., 0);
            self.history[N_FFT - 1] = sample;
            self.pending += 1;
            if self.pending == N_FFT {
                self.step(&mut out)?;
            }
        }
        Ok(out)
    }

    /// Pushes silence through the model so every sample given so far comes out, then resets.
    pub fn flush(&mut self) -> Result<Vec<f32>> {
        let owed = self.pending - HOP + if self.started { HOP } else { 0 };
        let mut out = self.process(&vec![0f32; 2 * HOP])?;
        out.truncate(owed);
        self.reset();
        Ok(out)
    }

    /// One STFT frame from `history`: analyse, enhance, overlap-add, emit one finished block.
    fn step(&mut self, out: &mut Vec<f32>) -> Result<()> {
        for ((t, x), w) in self.time.iter_mut().zip(&self.history).zip(&self.window) {
            *t = x * w;
        }
        self.fft
            .process_with_scratch(&mut self.time, &mut self.spectrum, &mut self.scratch)
            .map_err(|e| eyre!("FFT failed: {e}"))?;
        for ((c, m), p) in self.spectrum.iter().zip(&mut self.mag).zip(&mut self.pha) {
            *m = c.norm().ln_1p();
            *p = c.im.atan2(c.re);
        }

        let (mut mag, pha) = self.model.step(&self.mag, &self.pha);
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

        let norm = 1.0 / N_FFT as f32;
        for ((o, t), w) in self.overlap.iter_mut().zip(&self.time).zip(&self.window) {
            *o += t * norm * w;
        }
        if self.started {
            out.extend(
                self.overlap[..HOP]
                    .iter()
                    .zip(&self.envelope)
                    .map(|(o, e)| o / e),
            );
        }
        self.started = true;
        self.overlap.copy_within(HOP.., 0);
        self.overlap[HOP..].fill(0.0);
        self.pending = HOP;
        Ok(())
    }
}
