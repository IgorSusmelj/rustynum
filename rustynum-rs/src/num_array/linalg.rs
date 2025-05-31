//! # Linear Algebra Operations
//!
//! Provides operations such as matrix-vector multiplication using NumArray data structures.
use super::NumArray;
use std::iter::Sum;

use crate::simd_ops::SimdOps;
use crate::traits::{ExpLog, FromU32, FromUsize, NumOps};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Performs matrix-vector multiplication.
///
/// # Arguments
/// * `lhs` - A NumArray representing the left-hand side matrix.
/// * `rhs` - A NumArray representing the right-hand side vector.
///
/// # Returns
/// A NumArray containing the result of the multiplication.
///
/// # Panics
/// Panics if the number of columns in the matrix does not equal the length of the vector.
pub fn matrix_vector_multiply<T, Ops>(
    lhs: &NumArray<T, Ops>,
    rhs: &NumArray<T, Ops>,
) -> NumArray<T, Ops>
where
    T: Copy
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + Default
        + PartialOrd
        + FromU32
        + FromUsize
        + FromUsize
        + ExpLog
        + Neg<Output = T>
        + NumOps
        + Debug,
    Ops: SimdOps<T>,
{
    assert!(
        lhs.shape()[1] == rhs.shape()[0],
        "Column count of the matrix must match the length of the vector."
    );

    let rows = lhs.shape()[0];
    let mut result = NumArray::new(vec![T::default(); rows]);
    let rhs_data = rhs.get_data().to_vec();

    for i in 0..rows {
        let lhs_row = lhs.row_slice(i);
        let sum = Ops::dot_product(lhs_row, &rhs_data);
        result.set(&[i], sum);
    }

    result
}

/// Performs matrix-matrix multiplication.
///
/// # Arguments
/// * `lhs` - A NumArray representing the left-hand side matrix.
/// * `rhs` - A NumArray representing the right-hand side matrix.
///
/// # Returns
/// A NumArray containing the result of the multiplication.
///
/// # Panics
/// Panics if the number of columns in the left-hand side matrix does not equal the number of rows in the right-hand side matrix.
pub fn matrix_matrix_multiply<T, Ops>(
    lhs: &NumArray<T, Ops>,
    rhs: &NumArray<T, Ops>,
) -> NumArray<T, Ops>
where
    T: Copy
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + Default
        + PartialOrd
        + FromU32
        + FromUsize
        + FromUsize
        + ExpLog
        + Neg<Output = T>
        + NumOps
        + Debug,
    Ops: SimdOps<T>,
{
    assert!(
        lhs.shape()[1] == rhs.shape()[0],
        "Column count of the first matrix must match the row count of the second."
    );

    let rows = lhs.shape()[0];
    let cols = rhs.shape()[1];
    let inner_dim = lhs.shape()[1];

    let mut result_data = vec![T::default(); rows * cols];

    Ops::matrix_multiply(
        lhs.get_data(),
        rhs.get_data(),
        &mut result_data,
        rows,
        inner_dim,
        cols,
    );

    NumArray::new_with_shape(result_data, vec![rows, cols])
}

/// Convenience function that checks the shapes and performs the appropriate matrix multiplication.
///
/// # Arguments
/// * `lhs` - A NumArray representing the left-hand side matrix.
/// * `rhs` - A NumArray representing the right-hand side matrix or vector.
///
/// # Returns
/// A NumArray containing the result of the multiplication.
///
/// # Panics
/// Panics if the shapes are not compatible for either matrix-vector or matrix-matrix multiplication.
pub fn matrix_multiply<T, Ops>(lhs: &NumArray<T, Ops>, rhs: &NumArray<T, Ops>) -> NumArray<T, Ops>
where
    T: Copy
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + Default
        + PartialOrd
        + FromU32
        + FromUsize
        + FromUsize
        + ExpLog
        + Neg<Output = T>
        + NumOps
        + Debug,
    Ops: SimdOps<T>,
{
    match rhs.shape().len() {
        1 => matrix_vector_multiply(lhs, rhs),
        2 => matrix_matrix_multiply(lhs, rhs),
        _ => panic!("Unsupported shape for RHS; only vectors or matrices are supported."),
    }
}

#[cfg(test)]
mod tests {
    use crate::NumArrayF32;

    use super::*;

    #[test]
    fn test_matrix_vector_multiply_correct_calculation() {
        let matrix = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let vector = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0], vec![3]);
        let result = matrix_multiply(&matrix, &vector);

        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.get_data(), &[14.0, 32.0]);
    }

    #[test]
    #[should_panic(expected = "Column count of the matrix must match the length of the vector.")]
    fn test_matrix_vector_multiply_dimension_mismatch() {
        let matrix = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let vector = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

        let _result = matrix_multiply(&matrix, &vector);
    }

    #[test]
    fn test_matrix_vector_multiply_empty_vector() {
        let matrix = NumArrayF32::new_with_shape(vec![], vec![2, 0]);
        let vector = NumArrayF32::new_with_shape(vec![], vec![0]);

        let result = matrix_multiply(&matrix, &vector);

        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.get_data(), &[0.0, 0.0]);
    }

    #[test]
    fn test_matrix_vector_multiply_single_element() {
        let matrix = NumArrayF32::new_with_shape(vec![5.0], vec![1, 1]);
        let vector = NumArrayF32::new_with_shape(vec![2.0], vec![1]);

        let result = matrix_multiply(&matrix, &vector);

        assert_eq!(result.shape(), &[1]);
        assert_eq!(result.get_data(), &[10.0]);
    }

    #[test]
    fn test_matrix_matrix_multiply_size_16() {
        let matrix = NumArrayF32::new_with_shape(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0,
                2.0, 1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 14.0, 12.0, 10.0, 8.0, 6.0,
                4.0, 2.0, 0.0,
            ],
            vec![3, 16],
        );

        let matrix_rhs = NumArrayF32::new_with_shape(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            vec![16, 1],
        );

        let result = matrix_multiply(&matrix, &matrix_rhs);

        let expected_result = vec![1496.0, 816.0, 988.0];

        assert_eq!(result.shape(), &[3, 1]);
        assert_eq!(result.get_data(), &expected_result);
    }
}
