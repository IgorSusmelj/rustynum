use std::simd::f32x16;
use std::simd::f64x8;
use std::simd::num::SimdFloat;

const LANES_32: usize = 16;
const LANES_64: usize = 8;

/// SIMD-accelerated dot product for `f32`.
pub fn dot_product_simd_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut sum = f32x16::splat(0.0);

    let chunks = len / LANES_32;
    for i in 0..chunks {
        let a_simd = f32x16::from_slice(&a[i * LANES_32..]);
        let b_simd = f32x16::from_slice(&b[i * LANES_32..]);
        sum += a_simd * b_simd;
    }

    let mut scalar_sum = sum.reduce_sum();
    for i in (chunks * LANES_32)..len {
        scalar_sum += a[i] * b[i];
    }

    scalar_sum
}

/// SIMD-accelerated dot product for `f32`.
pub fn dot_product_simd_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());

    let (a_extra, a_chunks) = a.as_rchunks();
    let (b_extra, b_chunks) = b.as_rchunks();

    // These are always true, but for emphasis:
    assert_eq!(a_chunks.len(), b_chunks.len());
    assert_eq!(a_extra.len(), b_extra.len());

    let mut sums = [0.0; LANES_64];
    for ((x, y), d) in std::iter::zip(a_extra, b_extra).zip(&mut sums) {
        *d = x * y;
    }

    let mut sums = f64x8::from_array(sums);
    std::iter::zip(a_chunks, b_chunks).for_each(|(x, y)| {
        sums += f64x8::from_array(*x) * f64x8::from_array(*y);
    });

    sums.reduce_sum()
}
