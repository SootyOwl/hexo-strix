//! Plain-loop tensor ops for the hexo-infer forward pass.
//!
//! Weights use PyTorch `Linear` layout: `w` is row-major `(out_dim, in_dim)`,
//! `y = x @ w^T + b`. All math f32, matching the eager oracle.

/// y[n*out+o] = b[o] + Σ_i x[n*in+i] * w[o*in+i], for n rows.
///
/// The dot product runs on LANES independent accumulators so the compiler can
/// vectorize the reduction (strict-FP scalar chains can't be reassociated).
/// Summation order therefore differs from a naive loop; the parity fixtures
/// (1e-4 tiny / 1e-3 real) bound the drift.
pub fn linear(x: &[f32], n: usize, w: &[f32], b: &[f32], in_dim: usize, out_dim: usize, out: &mut Vec<f32>) {
    debug_assert_eq!(x.len(), n * in_dim, "linear: x stride");
    debug_assert_eq!(w.len(), out_dim * in_dim, "linear: w shape");
    debug_assert_eq!(b.len(), out_dim, "linear: b shape");
    const LANES: usize = 8;
    out.clear();
    out.resize(n * out_dim, 0.0);
    for row in 0..n {
        let xr = &x[row * in_dim..(row + 1) * in_dim];
        let or_ = &mut out[row * out_dim..(row + 1) * out_dim];
        for (o, out_v) in or_.iter_mut().enumerate() {
            let wr = &w[o * in_dim..(o + 1) * in_dim];
            let mut acc = [0.0f32; LANES];
            let mut xc = xr.chunks_exact(LANES);
            let mut wc = wr.chunks_exact(LANES);
            for (xk, wk) in (&mut xc).zip(&mut wc) {
                for l in 0..LANES {
                    acc[l] += xk[l] * wk[l];
                }
            }
            let mut s = b[o];
            for l in 0..LANES {
                s += acc[l];
            }
            for (xv, wv) in xc.remainder().iter().zip(wc.remainder()) {
                s += xv * wv;
            }
            *out_v = s;
        }
    }
}

/// PyTorch LayerNorm: per-row mean/population-variance over `dim`, eps 1e-5, affine.
pub fn layer_norm(x: &[f32], n: usize, dim: usize, gamma: &[f32], beta: &[f32], out: &mut Vec<f32>) {
    debug_assert_eq!(x.len(), n * dim, "layer_norm: x stride");
    const EPS: f32 = 1e-5;
    out.clear();
    out.resize(n * dim, 0.0);
    for row in 0..n {
        let xr = &x[row * dim..(row + 1) * dim];
        let mean = xr.iter().sum::<f32>() / dim as f32;
        let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        let or_ = &mut out[row * dim..(row + 1) * dim];
        for i in 0..dim {
            or_[i] = (xr[i] - mean) * inv * gamma[i] + beta[i];
        }
    }
}

pub fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_hand_computed() {
        // 1 row, 2 -> 2: w = [[1,2],[3,4]] (out,in), b = [10, 20], x = [5, 6]
        // y0 = 10 + 1*5 + 2*6 = 27 ; y1 = 20 + 3*5 + 4*6 = 59
        let mut out = Vec::new();
        linear(&[5.0, 6.0], 1, &[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0], 2, 2, &mut out);
        assert_eq!(out, vec![27.0, 59.0]);
    }

    #[test]
    fn linear_lanes_matches_naive() {
        // in_dim 11 = one full LANES chunk + remainder 3 (the real node_dim).
        let (n, in_dim, out_dim) = (3, 11, 5);
        let x: Vec<f32> = (0..n * in_dim).map(|i| ((i * 37 % 19) as f32 - 9.0) * 0.13).collect();
        let w: Vec<f32> = (0..out_dim * in_dim).map(|i| ((i * 53 % 23) as f32 - 11.0) * 0.07).collect();
        let b: Vec<f32> = (0..out_dim).map(|o| o as f32 * 0.5 - 1.0).collect();
        let mut out = Vec::new();
        linear(&x, n, &w, &b, in_dim, out_dim, &mut out);
        for row in 0..n {
            for o in 0..out_dim {
                let mut expect = b[o] as f64;
                for i in 0..in_dim {
                    expect += x[row * in_dim + i] as f64 * w[o * in_dim + i] as f64;
                }
                let got = out[row * out_dim + o] as f64;
                assert!((got - expect).abs() < 1e-5, "row {row} out {o}: {got} vs {expect}");
            }
        }
    }

    #[test]
    fn layer_norm_hand_computed() {
        // x = [1, 3]: mean 2, pop-var 1 -> normalized ±1/sqrt(1+1e-5)
        let mut out = Vec::new();
        layer_norm(&[1.0, 3.0], 1, 2, &[1.0, 1.0], &[0.0, 0.0], &mut out);
        let expect = 1.0 / (1.0f32 + 1e-5).sqrt();
        assert!((out[0] + expect).abs() < 1e-6, "{} vs {}", out[0], -expect);
        assert!((out[1] - expect).abs() < 1e-6);
    }

    #[test]
    fn relu_works() {
        let mut x = vec![-1.0, 0.0, 2.5];
        relu_inplace(&mut x);
        assert_eq!(x, vec![0.0, 0.0, 2.5]);
    }
}