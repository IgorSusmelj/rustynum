//! # Linear Algebra Operations
//!
//! Provides operations such as matrix-vector multiplication using NumArray data structures.
use super::num_array::{NumArray, NumArray32, NumArray64};
use std::iter::Sum;

use crate::simd_ops::SimdOps;
use crate::traits::{FromU32, FromUsize, NumOps};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

/// Performs matrix multiplication, handling both matrix-vector and matrix-matrix cases.
///
/// # Arguments
/// * `lhs` - A NumArray representing the left-hand side matrix.
/// * `rhs` - A NumArray representing the right-hand side matrix or vector.
///
/// # Returns
/// A NumArray containing the result of the multiplication.
///
/// # Panics
/// Panics if the number of columns in the matrix does not equal the length of the vector.
pub fn matrix_multiply<T, Ops>(lhs: &NumArray<T, Ops>, rhs: &NumArray<T, Ops>) -> NumArray<T, Ops>
where
    T: Copy
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + Default
        + FromU32
        + FromUsize
        + NumOps
        + Debug,
    Ops: SimdOps<T>,
{
    assert!(
        lhs.shape()[1] == rhs.shape()[0],
        "Column count of the first matrix must match the row count of the second."
    );

    let rows = lhs.shape()[0];
    let cols = if rhs.shape().len() > 1 {
        rhs.shape()[1]
    } else {
        1
    };

    let mut result = NumArray::new(vec![T::default(); rows * cols]);

    for j in 0..cols {
        let rhs_col = if rhs.shape().len() > 1 {
            rhs.column_slice(j)
        } else {
            rhs.get_data().to_vec()
        };
        for i in 0..rows {
            let lhs_row = lhs.row_slice(i);

            let sum = Ops::dot_product(lhs_row, &rhs_col);

            // Compute index for flat storage in result
            let index = i * cols + j;
            result.set(&[index], sum); // Make sure indexing respects flat storage
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_vector_multiply_correct_calculation() {
        let matrix = NumArray32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let vector = NumArray32::new_with_shape(vec![1.0, 2.0, 3.0], vec![3]);

        let result = matrix_multiply(&matrix, &vector);

        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.get_data(), &[14.0, 32.0]);
    }

    #[test]
    #[should_panic(
        expected = "Column count of the first matrix must match the row count of the second."
    )]
    fn test_matrix_vector_multiply_dimension_mismatch() {
        // Define a 2x3 matrix and a 4x1 vector (incorrect dimensions)
        let matrix = NumArray32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let vector = NumArray32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

        // This call should panic due to dimension mismatch
        let _result = matrix_multiply(&matrix, &vector);
    }

    #[test]
    fn test_matrix_vector_multiply_empty_vector() {
        // Define a 2x0 matrix (empty) and a 0x1 vector (empty)
        let matrix = NumArray32::new_with_shape(vec![], vec![2, 0]);
        let vector = NumArray32::new_with_shape(vec![], vec![0]);

        // Perform the matrix-vector multiplication
        let result = matrix_multiply(&matrix, &vector);

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
        let result = matrix_multiply(&matrix, &vector);

        // Check results
        // The expected output is a 1x1 vector where the element is 5*2 = 10
        assert_eq!(result.shape(), &[1]);
        assert_eq!(result.get_data(), &[10.0]);
    }
}
