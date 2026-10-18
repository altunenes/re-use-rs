//! Small dense kernels on flat row-major `f32` buffers.

/// `C[m×n] = op(A)[m×k] · op(B)[k×n] + beta·C`, row-major, like `cblas_sgemm`.
/// `op(X)` is `X` or its transpose; `lda`/`ldb`/`ldc` are row strides of the stored matrices.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    let a_len = if trans_a {
        (k - 1) * lda + m
    } else {
        (m - 1) * lda + k
    };
    let b_len = if trans_b {
        (n - 1) * ldb + k
    } else {
        (k - 1) * ldb + n
    };
    assert!(a.len() >= a_len && b.len() >= b_len && c.len() >= (m - 1) * ldc + n);
    imp::gemm(trans_a, trans_b, m, n, k, a, lda, b, ldb, beta, c, ldc);
}

#[cfg(target_os = "macos")]
mod imp {
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }

    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;
    const TRANS: i32 = 112;

    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    ) {
        let t = |x: bool| if x { TRANS } else { NO_TRANS };
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                t(trans_a),
                t(trans_b),
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                beta,
                c.as_mut_ptr(),
                ldc as i32,
            )
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    ) {
        let (rsa, csa) = if trans_a { (1, lda) } else { (lda, 1) };
        let (rsb, csb) = if trans_b { (1, ldb) } else { (ldb, 1) };
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                rsa as isize,
                csa as isize,
                b.as_ptr(),
                rsb as isize,
                csb as isize,
                beta,
                c.as_mut_ptr(),
                ldc as isize,
                1,
            )
        }
    }
}

/// `exp` accurate to about 1 ulp for `x` in [-87, 88], written so loops over it vectorize.
#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    // Adding 1.5·2^23 rounds to the nearest integer and leaves it in the low mantissa bits.
    let t = x * std::f32::consts::LOG2_E + 12_582_912.0;
    let n = t - 12_582_912.0;
    let r = x - n * 0.693_359_4 + n * 2.121_944_4e-4;
    let mut p = 1.987_569_1e-4_f32;
    p = p * r + 1.398_199_9e-3;
    p = p * r + 8.333_452e-3;
    p = p * r + 4.166_579_6e-2;
    p = p * r + 1.666_666_5e-1;
    p = p * r + 0.5;
    let p = p * r * r + r + 1.0;
    p * f32::from_bits(((t.to_bits() as i32 - 0x4B40_0000 + 127) as u32) << 23)
}

/// Natural log accurate to about 1 ulp for positive normal `x` (Cephes `logf`), vectorizable.
#[inline(always)]
pub fn fast_ln(x: f32) -> f32 {
    let bits = x.to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 126;
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f00_0000);
    let small = m < std::f32::consts::FRAC_1_SQRT_2;
    let e = (e - small as i32) as f32;
    let m = if small { m + m - 1.0 } else { m - 1.0 };
    let z = m * m;
    let mut y = 7.037_683_6e-2_f32;
    y = y * m - 1.151_461e-1;
    y = y * m + 1.167_699_9e-1;
    y = y * m - 1.242_014_1e-1;
    y = y * m + 1.424_932_3e-1;
    y = y * m - 1.666_805_8e-1;
    y = y * m + 2.000_071_5e-1;
    y = y * m - 2.499_999_4e-1;
    y = y * m + 3.333_333e-1;
    let y = y * m * z - 2.121_944_4e-4 * e - 0.5 * z;
    m + y + 0.693_359_4 * e
}

#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + fast_exp(-x))
}

/// `torch.nn.functional.softplus` (threshold 20).
#[inline(always)]
pub fn softplus(x: f32) -> f32 {
    let y = fast_exp(-x.abs());
    let u = 1.0 + y;
    let log1p = if u == 1.0 {
        y
    } else {
        fast_ln(u) * y / (u - 1.0)
    };
    let soft = x.max(0.0) + log1p;
    if x > 20.0 {
        x
    } else {
        soft
    }
}

/// LayerNorm over the last dimension of `rows × dim` data (eps 1e-5).
pub fn layer_norm_rows(x: &mut [f32], dim: usize, weight: &[f32], bias: &[f32]) {
    for row in x.chunks_exact_mut(dim) {
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for ((v, w), b) in row.iter_mut().zip(weight).zip(bias) {
            *v = (*v - mean) * inv * w + b;
        }
    }
}

/// LayerNorm over channels of channel-major `[channels][positions]` data, then per-channel PReLU.
pub fn layer_norm_prelu_channels(
    x: &mut [f32],
    channels: usize,
    weight: &[f32],
    bias: &[f32],
    prelu: &[f32],
) {
    let positions = x.len() / channels;
    let mut mean = vec![0f32; positions];
    let mut var = vec![0f32; positions];
    for row in x.chunks_exact(positions) {
        for (m, v) in mean.iter_mut().zip(row) {
            *m += v;
        }
    }
    mean.iter_mut().for_each(|m| *m /= channels as f32);
    for row in x.chunks_exact(positions) {
        for ((s, v), m) in var.iter_mut().zip(row).zip(&mean) {
            *s += (v - m) * (v - m);
        }
    }
    var.iter_mut()
        .for_each(|s| *s = 1.0 / (*s / channels as f32 + 1e-5).sqrt());
    for (c, row) in x.chunks_exact_mut(positions).enumerate() {
        let (w, b, a) = (weight[c], bias[c], prelu[c]);
        for ((v, m), inv) in row.iter_mut().zip(&mean).zip(&var) {
            let y = (*v - m) * inv * w + b;
            *v = if y > 0.0 { y } else { a * y };
        }
    }
}

/// Adds `bias[c]` to every position of channel `c` in channel-major data.
pub fn add_channel_bias(x: &mut [f32], bias: &[f32]) {
    let positions = x.len() / bias.len();
    for (row, b) in x.chunks_exact_mut(positions).zip(bias) {
        row.iter_mut().for_each(|v| *v += b);
    }
}

/// Transposes a `rows × cols` row-major matrix.
pub fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = x[r * cols + c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_exp_matches_std() {
        let mut worst = 0f32;
        let mut x = -80.0f32;
        while x < 80.0 {
            worst = worst.max((fast_exp(x) / x.exp() - 1.0).abs());
            x += 0.0137;
        }
        assert!(worst < 1e-6, "relative error {worst}");
    }

    #[test]
    fn fast_ln_and_softplus_match_std() {
        let mut worst = 0f32;
        let mut x = 1e-6f32;
        while x < 1e6 {
            worst = worst.max((fast_ln(x) - x.ln()).abs() / x.ln().abs().max(1.0));
            x *= 1.0137;
        }
        assert!(worst < 2e-7, "ln error {worst}");
        let mut worst = 0f32;
        let mut x = -30.0f32;
        while x < 30.0 {
            let exact = if x > 20.0 {
                x
            } else {
                (x as f64).exp().ln_1p() as f32
            };
            worst = worst.max((softplus(x) - exact).abs() / exact.max(1e-3));
            x += 0.0071;
        }
        assert!(worst < 1e-6, "softplus error {worst}");
    }

    #[test]
    fn gemm_transposes() {
        // A = [[1,2,3],[4,5,6]] (2×3), B = Aᵀ, A·Aᵀ = [[14,32],[32,77]]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = [0f32; 4];
        gemm(false, true, 2, 2, 3, &a, 3, &a, 3, 0.0, &mut c, 2);
        assert_eq!(c, [14.0, 32.0, 32.0, 77.0]);
        gemm(true, false, 3, 3, 2, &a, 3, &a, 3, 0.0, &mut [0f32; 9], 3);
    }
}
