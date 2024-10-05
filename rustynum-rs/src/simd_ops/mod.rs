use crate::helpers::parallel::parallel_for_chunks;
use std::simd::cmp::SimdOrd;
use std::simd::f32x16;
use std::simd::f64x8;
use std::simd::num::SimdFloat;
use std::simd::num::SimdUint;
use std::simd::u8x64;
use std::sync::Arc;
use std::sync::Mutex;

const LANES_8: usize = 64;
const LANES_32: usize = 16;
const LANES_64: usize = 8;

pub trait SimdOps<T> {
    fn matrix_multiply(a: &[T], b: &[T], c: &mut [T], m: usize, k: usize, n: usize);
    fn dot_product(a: &[T], b: &[T]) -> T;
    fn transpose(src: &[T], dst: &mut [T], n: usize, k: usize);
    fn sum(a: &[T]) -> T;
    fn min(a: &[T]) -> T;
    fn max(a: &[T]) -> T;
}

#[inline(always)]
fn dot_product_scalar<T>(a: &[T], b: &[T]) -> T
where
    T: std::ops::Mul<Output = T> + std::iter::Sum + Copy,
{
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

impl SimdOps<u8> for u8x64 {
    fn transpose(src: &[u8], dst: &mut [u8], rows: usize, cols: usize) {
        for i in 0..rows {
            for j in 0..cols {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    }

    fn matrix_multiply(a: &[u8], b: &[u8], c: &mut [u8], m: usize, k: usize, n: usize) {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(c.len(), m * n);

        c.fill(0);

        let mut b_transposed = vec![0u8; n * k];
        Self::transpose(b, &mut b_transposed, k, n);

        // We need to use a Mutex to safely write to the shared result matrix
        let c_shared = Arc::new(Mutex::new(c));

        parallel_for_chunks(0, m, |row_start, row_end| {
            // Clone the shared result matrix for each thread
            let c_lock = Arc::clone(&c_shared);

            for i in row_start..row_end {
                let a_row = &a[i * k..(i + 1) * k];
                let mut c_row = vec![0u8; n];

                for j in 0..n {
                    let b_col = &b_transposed[j * k..(j + 1) * k];
                    c_row[j] = Self::dot_product(a_row, b_col);
                }
                //
                let mut c_guard = c_lock.lock().unwrap();
                c_guard[i * n..(i + 1) * n].copy_from_slice(&c_row);
            }
        });
    }

    fn dot_product(a: &[u8], b: &[u8]) -> u8 {
        assert_eq!(a.len(), b.len());
        let len = a.len();

        let chunks = len / LANES_8;

        let mut sum1 = u8x64::splat(0);
        let mut sum2 = u8x64::splat(0);

        // Main loop with manual unrolling
        for i in (0..chunks).step_by(2) {
            let a1 = u8x64::from_slice(&a[i * LANES_8..]);
            let b1 = u8x64::from_slice(&b[i * LANES_8..]);
            sum1 += a1 * b1;

            if i + 1 < chunks {
                let a2 = u8x64::from_slice(&a[(i + 1) * LANES_8..]);
                let b2 = u8x64::from_slice(&b[(i + 1) * LANES_8..]);
                sum2 += a2 * b2;
            }
        }

        let mut scalar_sum = (sum1 + sum2).reduce_sum();

        // Efficient tail handling
        let remainder = len % LANES_8;
        if remainder > 0 {
            let tail_start = len - remainder;
            scalar_sum += dot_product_scalar(&a[tail_start..], &b[tail_start..]);
        }

        scalar_sum
    }

    fn sum(a: &[u8]) -> u8 {
        let mut sum = u8x64::splat(0);
        let chunks = a.len() / 64;

        for i in 0..chunks {
            let simd_chunk = u8x64::from_slice(&a[i * 64..]);
            sum += simd_chunk;
        }

        let mut scalar_sum = sum.reduce_sum();
        // Sum any remaining elements that didn't fit into a SIMD chunk
        for i in (chunks * LANES_8)..a.len() {
            scalar_sum += a[i];
        }

        scalar_sum
    }

    fn min(a: &[u8]) -> u8 {
        let simd_min_initial_value = u8::MAX;
        let mut simd_min = u8x64::splat(simd_min_initial_value);
        let chunks = a.len() / LANES_8;
        for i in 0..chunks {
            let simd_chunk = u8x64::from_slice(&a[i * LANES_8..]);
            simd_min = simd_min.simd_min(simd_chunk);
        }
        let mut final_min = simd_min.reduce_min();
        // Handle remaining elements
        for i in chunks * LANES_8..a.len() {
            final_min = final_min.min(a[i]);
        }
        final_min
    }

    fn max(a: &[u8]) -> u8 {
        let simd_max_initial_value = u8::MIN;
        let mut simd_max = u8x64::splat(simd_max_initial_value);
        let chunks = a.len() / LANES_8;
        for i in 0..chunks {
            let simd_chunk = u8x64::from_slice(&a[i * LANES_8..]);
            simd_max = simd_max.simd_max(simd_chunk);
        }
        let mut final_max = simd_max.reduce_max();
        // Handle remaining elements
        for i in chunks * LANES_8..a.len() {
            final_max = final_max.max(a[i]);
        }
        final_max
    }
}

impl SimdOps<f32> for f32x16 {
    fn transpose(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
        for i in 0..rows {
            for j in 0..cols {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    }

    fn matrix_multiply(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(c.len(), m * n);

        c.fill(0.0);

        let mut b_transposed = vec![0.0f32; n * k];
        Self::transpose(b, &mut b_transposed, k, n);

        // We need to use a Mutex to safely write to the shared result matrix
        let c_shared = Arc::new(Mutex::new(c));

        parallel_for_chunks(0, m, |row_start, row_end| {
            // Clone the shared result matrix for each thread
            let c_lock = Arc::clone(&c_shared);

            for i in row_start..row_end {
                let a_row = &a[i * k..(i + 1) * k];
                let mut c_row = vec![0.0; n];

                for j in 0..n {
                    let b_col = &b_transposed[j * k..(j + 1) * k];
                    c_row[j] = Self::dot_product(a_row, b_col);
                }
                //
                let mut c_guard = c_lock.lock().unwrap();
                c_guard[i * n..(i + 1) * n].copy_from_slice(&c_row);
            }
        });
    }

    fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let len = a.len();

        let chunks = len / LANES_32;

        let mut sum1 = f32x16::splat(0.0);
        let mut sum2 = f32x16::splat(0.0);

        // Main loop with manual unrolling
        for i in (0..chunks).step_by(2) {
            let a1 = f32x16::from_slice(&a[i * LANES_32..]);
            let b1 = f32x16::from_slice(&b[i * LANES_32..]);
            sum1 += a1 * b1;

            if i + 1 < chunks {
                let a2 = f32x16::from_slice(&a[(i + 1) * LANES_32..]);
                let b2 = f32x16::from_slice(&b[(i + 1) * LANES_32..]);
                sum2 += a2 * b2;
            }
        }

        let mut scalar_sum = (sum1 + sum2).reduce_sum();

        // Efficient tail handling
        let remainder = len % LANES_32;
        if remainder > 0 {
            let tail_start = len - remainder;
            scalar_sum += dot_product_scalar(&a[tail_start..], &b[tail_start..]);
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
    fn transpose(src: &[f64], dst: &mut [f64], rows: usize, cols: usize) {
        for i in 0..rows {
            for j in 0..cols {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    }

    fn matrix_multiply(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(c.len(), m * n);

        c.fill(0.0);

        let mut b_transposed = vec![0.0f64; n * k];
        Self::transpose(b, &mut b_transposed, k, n);

        // We need to use a Mutex to safely write to the shared result matrix
        let c_shared = Arc::new(Mutex::new(c));

        parallel_for_chunks(0, m, |row_start, row_end| {
            // Clone the shared result matrix for each thread
            let c_lock = Arc::clone(&c_shared);

            for i in row_start..row_end {
                let a_row = &a[i * k..(i + 1) * k];
                let mut c_row = vec![0.0; n];

                for j in 0..n {
                    let b_col = &b_transposed[j * k..(j + 1) * k];
                    c_row[j] = Self::dot_product(a_row, b_col);
                }
                //
                let mut c_guard = c_lock.lock().unwrap();
                c_guard[i * n..(i + 1) * n].copy_from_slice(&c_row);
            }
        });
    }

    fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let len = a.len();

        let chunks = len / LANES_64;

        let mut sum1 = f64x8::splat(0.0);
        let mut sum2 = f64x8::splat(0.0);

        // Main loop with manual unrolling
        for i in (0..chunks).step_by(2) {
            let a1 = f64x8::from_slice(&a[i * LANES_64..]);
            let b1 = f64x8::from_slice(&b[i * LANES_64..]);
            sum1 += a1 * b1;

            if i + 1 < chunks {
                let a2 = f64x8::from_slice(&a[(i + 1) * LANES_64..]);
                let b2 = f64x8::from_slice(&b[(i + 1) * LANES_64..]);
                sum2 += a2 * b2;
            }
        }

        let mut scalar_sum = (sum1 + sum2).reduce_sum();

        // Efficient tail handling
        let remainder = len % LANES_64;
        if remainder > 0 {
            let tail_start = len - remainder;
            scalar_sum += dot_product_scalar(&a[tail_start..], &b[tail_start..]);
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
    fn test_transpose_3x3_f32() {
        let src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let rows = 3;
        let cols = 3;
        let mut dst = vec![0.0f32; src.len()];

        f32x16::transpose(&src, &mut dst, rows, cols);

        let expected = [1.0f32, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];

        assert_eq!(
            dst, expected,
            "The transposed matrix does not match the expected result for a 3x3 matrix."
        );
    }

    #[test]
    fn test_dot_product_f32() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        let result = f32x16::dot_product(&a, &b);
        assert_eq!(result, 20.0);
    }

    #[test]
    fn test_dot_product_u8() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        let b = [8, 7, 6, 5, 4, 3, 2, 1];
        let result = u8x64::dot_product(&a, &b);
        assert_eq!(result, 120);
    }

    #[test]
    fn test_matrix_multiply_small_known_result_f32() {
        let a = vec![
            1.0, 2.0, 3.0, // 2x3 matrix
            4.0, 5.0, 6.0,
        ];
        let b = vec![
            7.0, 8.0, // 3x2 matrix
            9.0, 10.0, 11.0, 12.0,
        ];
        let mut c = vec![0.0; 4]; // Result will be 2x2 matrix

        let m = 2;
        let k = 3;
        let n = 2;

        f32x16::matrix_multiply(&a, &b, &mut c, m, k, n);

        let expected = vec![58.0, 64.0, 139.0, 154.0];
        assert_eq!(c, expected);
    }

    #[test]
    fn test_matrix_multiply_small_known_result_u8() {
        let a = vec![
            1, 2, 3, // 2x3 matrix
            4, 5, 6,
        ];
        let b = vec![
            7, 8, // 3x2 matrix
            9, 10, 11, 12,
        ];
        let mut c = vec![0; 4]; // Result will be 2x2 matrix

        let m = 2;
        let k = 3;
        let n = 2;

        u8x64::matrix_multiply(&a, &b, &mut c, m, k, n);

        let expected = vec![58, 64, 139, 154];
        assert_eq!(c, expected);
    }

    #[test]
    fn test_matrix_multiply_non_square_matrices_f32() {
        let a = vec![
            1.0, 4.0, // 3x2 matrix
            2.0, 5.0, 3.0, 6.0,
        ];
        let b = vec![
            7.0, 9.0, 11.0, // 2x3 matrix
            8.0, 10.0, 12.0,
        ];
        let mut c = vec![0.0; 9]; // Result will be 3x3 matrix

        let m = 3;
        let k = 2;
        let n = 3;

        f32x16::matrix_multiply(&a, &b, &mut c, m, k, n);

        let expected = vec![39.0, 49.0, 59.0, 54.0, 68.0, 82.0, 69.0, 87.0, 105.0];
        assert_eq!(c, expected);
    }

    #[test]
    fn test_matrix_multiply_negative_and_zero_values_f32() {
        let a = vec![
            0.0, -2.0, 3.0, // 2x3 matrix
            -4.0, 5.0, -6.0,
        ];
        let b = vec![
            -1.0, 0.0, // 3x2 matrix
            2.0, -3.0, 0.0, 4.0,
        ];
        let mut c = vec![0.0; 4]; // Result will be 2x2 matrix

        let m = 2;
        let k = 3;
        let n = 2;

        f32x16::matrix_multiply(&a, &b, &mut c, m, k, n);

        let expected = vec![-4.0, 18.0, 14.0, -39.0];
        assert_eq!(c, expected);
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
    fn test_transpose_3x3_f64() {
        let src = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let rows = 3;
        let cols = 3;
        let mut dst = vec![0.0f64; src.len()];

        f64x8::transpose(&src, &mut dst, rows, cols);

        let expected = [1.0f64, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];

        assert_eq!(
            dst, expected,
            "The transposed matrix does not match the expected result for a 3x3 matrix."
        );
    }

    #[test]
    fn test_dot_product_f64() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        let result = f64x8::dot_product(&a, &b);
        assert_eq!(result, 20.0);
    }

    #[test]
    fn test_matrix_multiply_small_known_result_f64() {
        let a = vec![
            1.0, 2.0, 3.0, // 2x3 matrix
            4.0, 5.0, 6.0,
        ];
        let b = vec![
            7.0, 8.0, // 3x2 matrix
            9.0, 10.0, 11.0, 12.0,
        ];
        let mut c = vec![0.0; 4]; // Result will be 2x2 matrix

        let m = 2;
        let k = 3;
        let n = 2;

        f64x8::matrix_multiply(&a, &b, &mut c, m, k, n);

        let expected = vec![58.0, 64.0, 139.0, 154.0];
        assert_eq!(c, expected);
    }

    #[test]
    fn test_matrix_multiply_non_square_matrices_f64() {
        let a = vec![
            1.0, 4.0, // 3x2 matrix
            2.0, 5.0, 3.0, 6.0,
        ];
        let b = vec![
            7.0, 9.0, 11.0, // 2x3 matrix
            8.0, 10.0, 12.0,
        ];
        let mut c = vec![0.0; 9]; // Result will be 3x3 matrix

        let m = 3;
        let k = 2;
        let n = 3;

        f64x8::matrix_multiply(&a, &b, &mut c, m, k, n);

        let expected = vec![39.0, 49.0, 59.0, 54.0, 68.0, 82.0, 69.0, 87.0, 105.0];
        assert_eq!(c, expected);
    }

    #[test]
    fn test_matrix_multiply_negative_and_zero_values_f64() {
        let a = vec![
            0.0, -2.0, 3.0, // 2x3 matrix
            -4.0, 5.0, -6.0,
        ];
        let b = vec![
            -1.0, 0.0, // 3x2 matrix
            2.0, -3.0, 0.0, 4.0,
        ];
        let mut c = vec![0.0; 4]; // Result will be 2x2 matrix

        let m = 2;
        let k = 3;
        let n = 2;

        f64x8::matrix_multiply(&a, &b, &mut c, m, k, n);

        let expected = vec![-4.0, 18.0, 14.0, -39.0];
        assert_eq!(c, expected);
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
