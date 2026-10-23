//! The offline model's TF-Mamba blocks on a GPU, with Burn. Each Mamba runs as three matmuls
//! and two CubeCL kernels: a causal depthwise conv, and the selective scan with its dt projection
//! and output gate, where one thread per (sequence, channel) walks the whole sequence.
//!
//! The real-time model stays on the CPU: one 20 ms frame per step gives the GPU little work per
//! launch, and waiting for every frame's result makes frame times uneven.

use crate::weights::Weights;
use burn::backend::{backend_extension, tensor::FloatTensor, Backend, Dispatch};
use burn::tensor::{Device, DeviceKind, Tensor, TensorData};
use burn_cubecl::{
    kernel::into_contiguous, ops::numeric::empty_device_dtype, tensor::CubeTensor, CubeBackend,
};
use cubecl::{calculate_cube_count_elemwise, prelude::CubeDim};
use eyre::Result;
use kernel::{conv_kernel, scan_kernel};

/// Which GPU API to run on. Each one needs its cargo feature (`wgpu`, `metal`, `vulkan`, `cuda`,
/// `rocm`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuDevice {
    /// wgpu on whatever the system offers: Metal, Vulkan or DX12.
    Wgpu,
    /// Metal with native MSL shaders (Apple).
    Metal,
    /// Vulkan with SPIR-V shaders.
    Vulkan,
    /// CUDA device with this index (NVIDIA).
    Cuda(usize),
    /// ROCm device with this index (AMD).
    Rocm(usize),
}

impl GpuDevice {
    fn device(self) -> Result<Device> {
        let missing =
            |feature: &str| eyre::eyre!("reuse-rs was built without the `{feature}` feature");
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu => Ok(Device::wgpu(DeviceKind::DefaultDevice)),
            #[cfg(feature = "metal")]
            Self::Metal => Ok(Device::metal(DeviceKind::DefaultDevice)),
            #[cfg(feature = "vulkan")]
            Self::Vulkan => Ok(Device::vulkan(DeviceKind::DefaultDevice)),
            #[cfg(feature = "cuda")]
            Self::Cuda(i) => Ok(Device::cuda(i)),
            #[cfg(feature = "rocm")]
            Self::Rocm(i) => Ok(Device::rocm(i)),
            #[allow(unreachable_patterns)]
            other => Err(missing(match other {
                Self::Wgpu => "wgpu",
                Self::Metal => "metal",
                Self::Vulkan => "vulkan",
                Self::Cuda(_) => "cuda",
                Self::Rocm(_) => "rocm",
            })),
        }
    }
}

#[backend_extension(Cube)]
pub trait MambaOps: Backend {
    /// Causal depthwise conv over each sequence, then silu. `xz` is `[batch·len, 2·d_inner]`
    /// (the conv reads its first half), `weight` is `[d_inner, taps]`. Returns
    /// `[batch·len, d_inner]`.
    fn mamba_conv(
        xz: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        batch: usize,
        reverse: bool,
    ) -> FloatTensor<Self>;

    /// Selective scan from zero state, with the dt projection, softplus and output gate fused:
    /// returns `(C·s + D·x)·silu(z)` as `[batch·len, d_inner]`. `dbl` rows are
    /// `[dt_raw | B | C]`, `dt_w` is `[d_inner, dt_rank]`, `a` is `[d_inner, 16]`.
    #[allow(clippy::too_many_arguments)]
    fn mamba_scan(
        xz: FloatTensor<Self>,
        xc: FloatTensor<Self>,
        dbl: FloatTensor<Self>,
        dt_w: FloatTensor<Self>,
        dt_b: FloatTensor<Self>,
        a: FloatTensor<Self>,
        d: FloatTensor<Self>,
        batch: usize,
        reverse: bool,
    ) -> FloatTensor<Self>;
}

mod kernel {
    use cubecl::prelude::*;

    /// One thread per (row, channel).
    #[cube(launch)]
    pub fn conv_kernel<F: Float>(
        xz: &Tensor<F>,
        weight: &Tensor<F>,
        bias: &Tensor<F>,
        xc: &mut Tensor<F>,
        rows: u32,
        len: u32,
        d_inner: u32,
        taps: u32,
        reverse: u32,
        #[define(F)] _dtype: ElemType,
    ) {
        let di = d_inner as usize;
        if ABSOLUTE_POS >= rows as usize * di {
            terminate!();
        }
        let row = ABSOLUTE_POS / di;
        let c = ABSOLUTE_POS % di;
        let taps = taps as usize;
        let len = len as usize;
        let l = row % len;
        let mut acc = bias[c];
        for k in 0..taps {
            // Tap k reads `taps - 1 - k` steps back; steps before the sequence start are zero.
            let back = taps - 1 - k;
            if reverse == 0 {
                if l >= back {
                    acc += weight[c * taps + k] * xz[(row - back) * 2 * di + c];
                }
            } else if l + back < len {
                acc += weight[c * taps + k] * xz[(row + back) * 2 * di + c];
            }
        }
        xc[row * di + c] = acc / (F::new(1.0f32) + (-acc).exp());
    }

    /// One thread per (sequence, channel), walking the whole sequence.
    #[cube(launch)]
    pub fn scan_kernel<F: Float>(
        xz: &Tensor<F>,
        xc: &Tensor<F>,
        dbl: &Tensor<F>,
        dt_w: &Tensor<F>,
        dt_b: &Tensor<F>,
        a: &Tensor<F>,
        d: &Tensor<F>,
        y: &mut Tensor<F>,
        batch: u32,
        len: u32,
        d_inner: u32,
        dt_rank: u32,
        reverse: u32,
        #[define(F)] _dtype: ElemType,
    ) {
        let di = d_inner as usize;
        if ABSOLUTE_POS >= batch as usize * di {
            terminate!();
        }
        let seq = ABSOLUTE_POS / di;
        let c = ABSOLUTE_POS % di;
        let (len, rank) = (len as usize, dt_rank as usize);
        let nd = rank + 32;
        let mut s = Array::new(16usize);
        let mut ac = Array::new(16usize);
        #[unroll]
        for n in 0..16usize {
            s[n] = F::new(0.0f32);
            ac[n] = a[c * 16 + n];
        }
        let (bias, skip) = (dt_b[c], d[c]);
        for i in 0..len {
            let mut l = i;
            if reverse != 0 {
                l = len - 1 - i;
            }
            let row = seq * len + l;
            let mut raw = bias;
            for k in 0..rank {
                raw += dbl[row * nd + k] * dt_w[c * rank + k];
            }
            // torch softplus, threshold 20.
            let mut delta = raw;
            if raw <= F::new(20.0f32) {
                delta = raw.exp().log1p();
            }
            let x = xc[row * di + c];
            let dx = delta * x;
            let mut acc = F::new(0.0f32);
            #[unroll]
            for n in 0..16usize {
                let decay = (delta * ac[n]).exp();
                s[n] = decay * s[n] + dx * dbl[row * nd + rank + n];
                acc += s[n] * dbl[row * nd + rank + 16 + n];
            }
            let z = xz[row * 2 * di + di + c];
            y[row * di + c] = (acc + skip * x) * (z / (F::new(1.0f32) + (-z).exp()));
        }
    }
}

const BLOCK: u32 = 256;

impl MambaOps for CubeBackend {
    fn mamba_conv(
        xz: CubeTensor,
        weight: CubeTensor,
        bias: CubeTensor,
        batch: usize,
        reverse: bool,
    ) -> CubeTensor {
        let (xz, weight, bias) = (
            into_contiguous(xz),
            into_contiguous(weight),
            into_contiguous(bias),
        );
        let rows = xz.meta.shape()[0];
        let (di, taps) = (weight.meta.shape()[0], weight.meta.shape()[1]);
        let xc = empty_device_dtype(
            xz.client.clone(),
            xz.device.clone(),
            [rows, di].into(),
            xz.dtype,
        );
        let threads = rows * di;
        let dtype = burn::backend::cubecl::dtype_to_storage_type(xz.dtype);
        let client = xz.client.clone();
        conv_kernel::launch(
            &client,
            calculate_cube_count_elemwise(&client, threads, CubeDim::new_1d(BLOCK)),
            CubeDim::new_1d(BLOCK),
            xz.into_tensor_arg(),
            weight.into_tensor_arg(),
            bias.into_tensor_arg(),
            xc.clone().into_tensor_arg(),
            rows as u32,
            (rows / batch) as u32,
            di as u32,
            taps as u32,
            reverse as u32,
            dtype,
        );
        xc
    }

    fn mamba_scan(
        xz: CubeTensor,
        xc: CubeTensor,
        dbl: CubeTensor,
        dt_w: CubeTensor,
        dt_b: CubeTensor,
        a: CubeTensor,
        d: CubeTensor,
        batch: usize,
        reverse: bool,
    ) -> CubeTensor {
        let [xz, xc, dbl, dt_w, dt_b, a, d] = [xz, xc, dbl, dt_w, dt_b, a, d].map(into_contiguous);
        let (rows, di) = (xc.meta.shape()[0], xc.meta.shape()[1]);
        let rank = dt_w.meta.shape()[1];
        let y = empty_device_dtype(
            xc.client.clone(),
            xc.device.clone(),
            [rows, di].into(),
            xc.dtype,
        );
        let threads = batch * di;
        let dtype = burn::backend::cubecl::dtype_to_storage_type(xc.dtype);
        let client = xc.client.clone();
        scan_kernel::launch(
            &client,
            calculate_cube_count_elemwise(&client, threads, CubeDim::new_1d(BLOCK)),
            CubeDim::new_1d(BLOCK),
            xz.into_tensor_arg(),
            xc.into_tensor_arg(),
            dbl.into_tensor_arg(),
            dt_w.into_tensor_arg(),
            dt_b.into_tensor_arg(),
            a.into_tensor_arg(),
            d.into_tensor_arg(),
            y.clone().into_tensor_arg(),
            batch as u32,
            (rows / batch) as u32,
            di as u32,
            rank as u32,
            reverse as u32,
            dtype,
        );
        y
    }
}

fn tensor<const D: usize>(w: &Weights, name: &str, device: &Device) -> Result<Tensor<D>> {
    let shape: [usize; D] = w
        .shape(name)?
        .try_into()
        .map_err(|_| eyre::eyre!("{name}: expected {D} dimensions"))?;
    Ok(Tensor::from_data(
        TensorData::new(w.get(name)?, shape),
        device,
    ))
}

/// `[out, in]` weight as `[in, out]`, for `x.matmul(w)`.
fn linear(w: &Weights, name: &str, device: &Device) -> Result<Tensor<2>> {
    Ok(tensor::<2>(w, name, device)?.transpose())
}

struct Mamba {
    in_proj: Tensor<2>,
    conv_w: Tensor<2>,
    conv_b: Tensor<1>,
    x_proj: Tensor<2>,
    dt_w: Tensor<2>,
    dt_b: Tensor<1>,
    a: Tensor<2>,
    d: Tensor<1>,
    out_proj: Tensor<2>,
}

impl Mamba {
    fn load(w: &Weights, prefix: &str, device: &Device) -> Result<Self> {
        let conv_w = tensor::<3>(w, &format!("{prefix}.conv1d.weight"), device)?;
        let [di, _, taps] = conv_w.dims();
        Ok(Self {
            in_proj: linear(w, &format!("{prefix}.in_proj.weight"), device)?,
            conv_w: conv_w.reshape([di, taps]),
            conv_b: tensor(w, &format!("{prefix}.conv1d.bias"), device)?,
            x_proj: linear(w, &format!("{prefix}.x_proj.weight"), device)?,
            dt_w: tensor(w, &format!("{prefix}.dt_proj.weight"), device)?,
            dt_b: tensor(w, &format!("{prefix}.dt_proj.bias"), device)?,
            a: tensor::<2>(w, &format!("{prefix}.A_log"), device)?
                .exp()
                .neg(),
            d: tensor(w, &format!("{prefix}.D"), device)?,
            out_proj: linear(w, &format!("{prefix}.out_proj.weight"), device)?,
        })
    }

    /// `x` is `[batch·len, d_model]`, sequence-major.
    fn forward(&self, x: Tensor<2>, batch: usize, reverse: bool) -> Tensor<2> {
        let xz = x.matmul(self.in_proj.clone());
        let xc = Tensor::<2>::from_dispatch(<Dispatch as MambaOps>::mamba_conv(
            xz.clone().into_dispatch(),
            self.conv_w.clone().into_dispatch(),
            self.conv_b.clone().into_dispatch(),
            batch,
            reverse,
        ));
        let dbl = xc.clone().matmul(self.x_proj.clone());
        let y = Tensor::<2>::from_dispatch(<Dispatch as MambaOps>::mamba_scan(
            xz.into_dispatch(),
            xc.into_dispatch(),
            dbl.into_dispatch(),
            self.dt_w.clone().into_dispatch(),
            self.dt_b.clone().into_dispatch(),
            self.a.clone().into_dispatch(),
            self.d.clone().into_dispatch(),
            batch,
            reverse,
        ));
        y.matmul(self.out_proj.clone())
    }
}

/// LayerNorm over the last dimension (eps 1e-5).
fn layer_norm(x: Tensor<2>, weight: &Tensor<1>, bias: &Tensor<1>) -> Tensor<2> {
    let mean = x.clone().mean_dim(1);
    let centered = x - mean;
    let var = centered.clone().square().mean_dim(1);
    centered / (var + 1e-5).sqrt() * weight.clone().unsqueeze::<2>() + bias.clone().unsqueeze::<2>()
}

/// Bidirectional MambaBlock: `norm(output_proj([fw(x) + x, flip(bw(flip x)) + x]))`. The
/// backward Mamba walks each sequence from its end, and `output_proj` is split into the halves
/// that multiply each direction.
struct BiMamba {
    forward: Mamba,
    backward: Mamba,
    proj_fw: Tensor<2>,
    proj_bw: Tensor<2>,
    output_bias: Tensor<1>,
    norm_w: Tensor<1>,
    norm_b: Tensor<1>,
}

impl BiMamba {
    fn load(w: &Weights, prefix: &str, device: &Device) -> Result<Self> {
        let proj = linear(w, &format!("{prefix}.output_proj.weight"), device)?;
        let [inp, out] = proj.dims();
        Ok(Self {
            forward: Mamba::load(w, &format!("{prefix}.forward_blocks"), device)?,
            backward: Mamba::load(w, &format!("{prefix}.backward_blocks"), device)?,
            proj_fw: proj.clone().slice([0..inp / 2, 0..out]),
            proj_bw: proj.slice([inp / 2..inp, 0..out]),
            output_bias: tensor(w, &format!("{prefix}.output_proj.bias"), device)?,
            norm_w: tensor(w, &format!("{prefix}.norm.weight"), device)?,
            norm_b: tensor(w, &format!("{prefix}.norm.bias"), device)?,
        })
    }

    fn forward(&self, x: Tensor<2>, batch: usize) -> Tensor<2> {
        let fw = self.forward.forward(x.clone(), batch, false) + x.clone();
        let bw = self.backward.forward(x.clone(), batch, true) + x;
        let out = fw.matmul(self.proj_fw.clone())
            + bw.matmul(self.proj_bw.clone())
            + self.output_bias.clone().unsqueeze::<2>();
        layer_norm(out, &self.norm_w, &self.norm_b)
    }
}

struct Block {
    time: BiMamba,
    freq: BiMamba,
}

/// The TF-Mamba blocks on a GPU. The convolutional encoder and decoders stay on the CPU, where
/// they run faster than Burn's conv2d at these sizes.
pub(crate) struct Model {
    blocks: Vec<Block>,
    device: Device,
}

impl Model {
    pub(crate) fn load(w: &Weights, blocks: usize, device: GpuDevice) -> Result<Self> {
        let device = device.device()?;
        let blocks = (0..blocks)
            .map(|i| {
                Ok(Block {
                    time: BiMamba::load(w, &format!("TSMamba.{i}.time_mamba"), &device)?,
                    freq: BiMamba::load(w, &format!("TSMamba.{i}.freq_mamba"), &device)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self { blocks, device })
    }

    /// `rows` is `[freq][time][channels]`; returns the same layout.
    pub(crate) fn mamba(
        &self,
        rows: &[f32],
        time: usize,
        freq: usize,
        channels: usize,
    ) -> Vec<f32> {
        let positions = time * freq;
        let mut x: Tensor<2> = Tensor::from_data(
            TensorData::new(rows.to_vec(), [positions, channels]),
            &self.device,
        );
        // [a][b][c] -> [b][a][c]
        let swap = |t: Tensor<2>, a: usize, b: usize| {
            t.reshape([a, b, channels])
                .swap_dims(0, 1)
                .reshape([positions, channels])
        };
        for block in &self.blocks {
            x = x.clone() + block.time.forward(x, freq);
            x = swap(x, freq, time);
            x = x.clone() + block.freq.forward(x, time);
            x = swap(x, time, freq);
        }
        x.into_data()
            .try_into_vec::<f32>()
            .expect("GPU output is f32")
    }
}
