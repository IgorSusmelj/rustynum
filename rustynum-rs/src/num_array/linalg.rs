//! # Linear Algebra Operations
//!
//! Provides operations such as matrix-vector multiplication using NumArray data structures.
use std::fmt::Debug;

use super::num_array::{NumArray, NumArray32, NumArray64};
use crate::simd_ops::SimdOps;
use crate::traits::{FromU32, FromUsize, NumOps};

/// Performs matrix-vector multiplication using NumArray.
///
/// # Arguments
/// * `matrix` - A NumArray representing the 2D matrix.
/// * `vector` - A NumArray representing the vector to multiply.
///
/// # Returns
/// A NumArray containing the result of the multiplication.
///
/// # Panics
/// Panics if the number of columns in the matrix does not equal the length of the vector.
pub fn matrix_vector_multiply<T, Ops>(
    matrix: &NumArray<T, Ops>,
    vector: &NumArray<T, Ops>,
) -> NumArray<T, Ops>
where
    T: Copy
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
    assert!(
        matrix.shape()[1] == vector.shape()[0],
        "Matrix and vector dimensions must match for multiplication."
    );

    let rows = matrix.shape()[0];
    let cols = matrix.shape()[1];

    let mut result = NumArray::new(vec![T::default(); rows]);

    for i in 0..rows {
        let mut sum = T::default(); // Default initialization
        for j in 0..cols {
            sum = sum + matrix.get(&[i, j]) * vector.get(&[j]);
        }
        result.set(&[i], sum);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_vector_multiply_correct_calculation() {
        // Define a 2x3 matrix and a 3x1 vector
        let matrix = NumArray32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let vector = NumArray32::new_with_shape(vec![1.0, 2.0, 3.0], vec![3]);

        // Perform the matrix-vector multiplication
        let result = matrix_vector_multiply(&matrix, &vector);

        // Check results
        // The expected output is a 2x1 vector where:
        // - The first element is 1*1 + 2*2 + 3*3 = 14
        // - The second element is 4*1 + 5*2 + 6*3 = 32
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.get_data(), &[14.0, 32.0]);
    }

    #[test]
    #[should_panic(expected = "Matrix and vector dimensions must match for multiplication.")]
    fn test_matrix_vector_multiply_dimension_mismatch() {
        // Define a 2x3 matrix and a 4x1 vector (incorrect dimensions)
        let matrix = NumArray32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let vector = NumArray32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

        // This call should panic due to dimension mismatch
        let _result = matrix_vector_multiply(&matrix, &vector);
    }

    #[test]
    fn test_matrix_vector_multiply_empty_vector() {
        // Define a 2x0 matrix (empty) and a 0x1 vector (empty)
        let matrix = NumArray32::new_with_shape(vec![], vec![2, 0]);
        let vector = NumArray32::new_with_shape(vec![], vec![0]);

        // Perform the matrix-vector multiplication
        let result = matrix_vector_multiply(&matrix, &vector);

        // Check results
        // The expected output is a 2x1 vector with default values (0.0)
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.get_data(), &[0.0, 0.0]);
    }

    #[test]
    fn test_matrix_vector_multiply_single_element() {
        // Define a 1x1 matrix and a 1x1 vector
        let matrix = NumArray32::new_with_shape(vec![5.0], vec![1, 1]);
        let vector = NumArray32::new_with_shape(vec![2.0], vec![1]);

        // Perform the matrix-vector multiplication
        let result = matrix_vector_multiply(&matrix, &vector);

        // Check results
        // The expected output is a 1x1 vector where the element is 5*2 = 10
        assert_eq!(result.shape(), &[1]);
        assert_eq!(result.get_data(), &[10.0]);
    }
}
