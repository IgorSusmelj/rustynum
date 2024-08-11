use std::simd::f32x16;
use std::simd::f64x8;
use std::simd::num::SimdFloat;
use std::simd::StdFloat;

const LANES_32: usize = 16;
const LANES_64: usize = 8;

pub trait SimdOps<T> {
    fn dot_product(a: &[T], b: &[T]) -> T;
    fn sum(a: &[T]) -> T;
    fn min(a: &[T]) -> T;
    fn max(a: &[T]) -> T;
}

impl SimdOps<f32> for f32x16 {
    fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let len = a.len();
        let sum = f32x16::splat(0.0);

        let chunks = len / LANES_32;
        for i in 0..chunks {
            let a_simd = f32x16::from_slice(&a[i * LANES_32..]);
            let b_simd = f32x16::from_slice(&b[i * LANES_32..]);
            let _ = sum.mul_add(a_simd, b_simd);
        }

        let mut scalar_sum = sum.reduce_sum();
        for i in (chunks * LANES_32)..len {
            scalar_sum += a[i] * b[i];
        }

        scalar_sum
    }

    fn sum(a: &[f32]) -> f32 {
        let mut sum = f32x16::splat(0.0);
        let chunks = a.len() / LANES_32;

        for i in 0..chunks {
            let simd_chunk = f32x16::from_slice(&a[i * LANES_32..]);
            sum += simd_chunk;
        }

        let mut scalar_sum = sum.reduce_sum();
        // Sum any remaining elements that didn't fit into a SIMD chunk
        for i in (chunks * LANES_32)..a.len() {
            scalar_sum += a[i];
        }

        scalar_sum
    }

    fn min(a: &[f32]) -> f32 {
        let simd_min_initial_value = f32::MAX;
        let mut simd_min = f32x16::splat(simd_min_initial_value);
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let simd_chunk = f32x16::from_slice(&a[i * LANES_32..]);
            simd_min = simd_min.simd_min(simd_chunk);
        }
        let mut final_min = simd_min.reduce_min();
        // Handle remaining elements
        for i in chunks * LANES_32..a.len() {
            final_min = final_min.min(a[i]);
        }
        final_min
    }

    fn max(a: &[f32]) -> f32 {
        let simd_max_initial_value = f32::MIN;
        let mut simd_max = f32x16::splat(simd_max_initial_value);
        let chunks = a.len() / LANES_32;
        for i in 0..chunks {
            let simd_chunk = f32x16::from_slice(&a[i * LANES_32..]);
            simd_max = simd_max.simd_max(simd_chunk);
        }
        let mut final_max = simd_max.reduce_max();
        // Handle remaining elements
        for i in chunks * LANES_32..a.len() {
            final_max = final_max.max(a[i]);
        }
        final_max
    }
}

impl SimdOps<f64> for f64x8 {
    fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let len = a.len();
        let sum = f64x8::splat(0.0);

        let chunks = len / LANES_64;
        for i in 0..chunks {
            let a_simd = f64x8::from_slice(&a[i * LANES_64..]);
            let b_simd = f64x8::from_slice(&b[i * LANES_64..]);
            let _ = sum.mul_add(a_simd, b_simd);
        }

        let mut scalar_sum = sum.reduce_sum();
        for i in (chunks * LANES_64)..len {
            scalar_sum += a[i] * b[i];
        }

        scalar_sum
    }

    fn sum(a: &[f64]) -> f64 {
        let mut sum = f64x8::splat(0.0);
        let chunks = a.len() / LANES_64;

        for i in 0..chunks {
            let simd_chunk = f64x8::from_slice(&a[i * LANES_64..]);
            sum += simd_chunk;
        }

        let mut scalar_sum = sum.reduce_sum();
        // Sum any remaining elements that didn't fit into a SIMD chunk
        for i in (chunks * LANES_64)..a.len() {
            scalar_sum += a[i];
        }

        scalar_sum
    }

    fn min(a: &[f64]) -> f64 {
        let simd_min_initial_value = f64::MAX;
        let mut simd_min = f64x8::splat(simd_min_initial_value);
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let simd_chunk = f64x8::from_slice(&a[i * LANES_64..]);
            simd_min = simd_min.simd_min(simd_chunk);
        }
        let mut final_min = simd_min.reduce_min();
        // Handle remaining elements
        for i in chunks * LANES_64..a.len() {
            final_min = final_min.min(a[i]);
        }
        final_min
    }

    fn max(a: &[f64]) -> f64 {
        let simd_max_initial_value = f64::MIN;
        let mut simd_max = f64x8::splat(simd_max_initial_value);
        let chunks = a.len() / LANES_64;
        for i in 0..chunks {
            let simd_chunk = f64x8::from_slice(&a[i * LANES_64..]);
            simd_max = simd_max.simd_max(simd_chunk);
        }
        let mut final_max = simd_max.reduce_max();
        // Handle remaining elements
        for i in chunks * LANES_64..a.len() {
            final_max = final_max.max(a[i]);
        }
        final_max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_f32() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        let result = f32x16::dot_product(&a, &b);
        assert_eq!(result, 20.0);
    }

    #[test]
    fn test_sum_f32() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let result = f32x16::sum(&a);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_min_f32() {
        let a = [4.0, 1.0, 3.0, 2.0];
        let result = f32x16::min(&a);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_max_f32() {
        let a = [4.0, 1.0, 3.0, 2.0];
        let result = f32x16::max(&a);
        assert_eq!(result, 4.0);
    }

    #[test]
    fn test_dot_product_f64() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        let result = f64x8::dot_product(&a, &b);
        assert_eq!(result, 20.0);
    }

    #[test]
    fn test_sum_f64() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let result = f64x8::sum(&a);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_min_f64() {
        let a = [4.0, 1.0, 3.0, 2.0];
        let result = f64x8::min(&a);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_max_f64() {
        let a = [4.0, 1.0, 3.0, 2.0];
        let result = f64x8::max(&a);
        assert_eq!(result, 4.0);
    }
}
