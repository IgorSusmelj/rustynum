use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};
use std::simd::{f32x16, f64x8};
use std::fmt::Debug;

use crate::simd_ops::SimdOps;
use crate::traits::{FromU32, FromUsize, NumOps};

pub type NumArray32 = NumArray<f32, f32x16>;
pub type NumArray64 = NumArray<f64, f64x8>;

pub struct NumArray<T, Ops> 
where
    T: Debug
{
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    _ops: PhantomData<Ops>,
}

impl<T, Ops> NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + NumOps
        + Copy
        + FromU32
        + FromUsize
        + Debug,
    Ops: SimdOps<T>,
{
    // Existing new method that requires shape to be specified
    pub fn new_with_shape(data: Vec<T>, shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
            _ops: PhantomData,
        }
    }

    pub fn new(data: Vec<T>) -> Self {
        let shape = vec![data.len()]; // Infer shape as 1D
        let strides = vec![1]; // For a 1D array, stride is always 1
        Self {
            data,
            shape,
            strides,
            _ops: PhantomData,
        }
    }

    pub fn get_data(&self) -> &Vec<T> {
        &self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn reshape(&mut self, new_shape: Vec<usize>) {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(
            new_size,
            self.data.len(),
            "New shape must be compatible with data size."
        );
        self.shape = new_shape;
        self.strides = Self::compute_strides(&self.shape);
    }

    pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = Vec::with_capacity(shape.len());
        let mut stride = 1;
        for &dim in shape.iter().rev() {
            strides.push(stride);
            stride *= dim;
        }
        strides.reverse();
        strides
    }

    pub fn dot(&self, other: &Self) -> T {
        Ops::dot_product(&self.data, &other.data)
    }

    pub fn mean(&self) -> T {
        let sum: T = Ops::sum(&self.data);
        let count = T::from_u32(self.data.len() as u32);
        sum / count
    }

    pub fn calculate_reduced_index(&self, index: usize, reduction_shape: &[usize]) -> usize {
        let mut reduced_index = 0;
        let mut stride = 1;
        let mut accumulated_strides = 1; // To handle collapsed dimensions correctly

        for (i, (&original_dim, &reduced_dim)) in self.shape.iter().zip(reduction_shape.iter()).rev().enumerate() {
            let index_in_dim = (index / stride) % original_dim;
            
            if reduced_dim != 1 {
                reduced_index += index_in_dim * accumulated_strides;
                accumulated_strides *= original_dim;  // Update only when not collapsing
            }

            stride *= original_dim;  // Always update stride to traverse dimensions

        }

        reduced_index
    }

    pub fn mean_axes(&self, axes: &[usize]) -> NumArray<T, Ops> {
        for &axis in axes {
            assert!(axis < self.shape.len(), "Axis {} out of bounds.", axis);
        }
    
        let mut reduced_shape = self.shape.clone();
        let mut total_elements_to_reduce = 1;

        for &axis in axes {
            total_elements_to_reduce *= self.shape[axis];
            reduced_shape[axis] = 1;  // Marking this axis for reduction
        }
        
        let reduced_size: usize = reduced_shape.iter().product();
        let mut reduced_data = vec![T::from_u32(0); reduced_size];
    
        // Process each element in the data
        for (i, &val) in self.data.iter().enumerate() {
            let reduced_idx = self.calculate_reduced_index(i, &reduced_shape);
            reduced_data[reduced_idx] = reduced_data[reduced_idx] + val;
        }
    
        // Divide each element in reduced_data by the number of elements that contributed to it
        for val in reduced_data.iter_mut() {
            *val = *val / T::from_usize(total_elements_to_reduce);
        }
    
        NumArray::new_with_shape(reduced_data, reduced_shape)
    }
    
    pub fn min(&self) -> T {
        Ops::min(&self.data)
    }

    pub fn max(&self) -> T {
        Ops::max(&self.data)
    }

    pub fn normalize(&self) -> Self {
        let norm_squared: T = self.data.iter().fold(T::from_u32(0), |acc, &x| acc + x * x);
        let norm = norm_squared.sqrt();
        let normalized_data = self.data.iter().map(|&x| x / norm).collect();
        Self {
            data: normalized_data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            _ops: PhantomData,
        }
    }

    pub fn slice(&self, start: usize, end: usize) -> Self {
        assert!(start <= end, "Start index must not exceed end index.");
        assert!(
            end <= self.data.len(),
            "End index must not exceed data length."
        );
        let slice_length = end - start;
        // For a simple 1D slice, the shape is just the length of the slice
        let new_shape = vec![slice_length];
        let sliced_data = self.data[start..end].to_vec();
        Self {
            data: sliced_data,
            shape: new_shape,
            // Strides for a 1D slice default to {1} since it's a linear slice
            strides: vec![1],
            _ops: PhantomData,
        }
    }
}

impl<T, Ops> From<Vec<T>> for NumArray<T, Ops>
where
    T: NumOps
        + Clone
        + Copy
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + Debug,
    Ops: SimdOps<T> + Default, // Ensure Ops can be defaulted or appropriately initialized
{
    fn from(data: Vec<T>) -> Self {
        let shape = vec![data.len()]; // Assume 1D shape based on the length of the data
        Self {
            data,
            shape,
            strides: vec![1], // Stride is 1 for a 1D array
            _ops: PhantomData,
        }
    }
}

impl<T, Ops> Clone for NumArray<T, Ops>
where
    T: Clone
    + Debug,
{
    fn clone(&self) -> Self {
        NumArray {
            data: self.data.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            _ops: PhantomData,
        }
    }
}

impl<'a, T, Ops> From<&'a [T]> for NumArray<T, Ops>
where
    T: 'a
        + NumOps
        + Clone
        + Copy
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        + Debug,
    Ops: SimdOps<T> + Default,
{
    fn from(data: &'a [T]) -> Self {
        let shape = vec![data.to_vec().len()]; // Assume 1D shape based on the length of the data
        Self {
            data: data.to_vec(),
            shape,
            strides: vec![1], // Stride is 1 for a 1D array
            _ops: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_shape_of_new_array() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let array = NumArray32::new(data.clone()); // Using NumArray32 for simplicity
        let expected_shape = vec![4]; // Expected shape for a 1D array with 4 elements
        assert_eq!(array.shape(), expected_shape.as_slice());
    }

    #[test]
    fn test_reshape_successfully() {
        let mut array = NumArray32::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let new_shape = vec![2, 3]; // New shape compatible with the size of data (2*3 = 6)
        array.reshape(new_shape.clone());
        assert_eq!(array.shape(), new_shape.as_slice());
    }

    #[test]
    fn test_reshape_and_strides() {
        let mut array = NumArray32::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let new_shape = vec![2, 4]; // Reshape to a 2x4 matrix
        array.reshape(new_shape.clone());
        assert_eq!(array.shape(), new_shape.as_slice());
        // Check strides for a 2x4 matrix
        let expected_strides = vec![4, 1]; // Moving to the next row jumps 4 elements, column jump is 1
        assert_eq!(array.strides, expected_strides);
    }

    #[test]
    fn test_dot_product_f32() {
        let a = NumArray32::new(vec![1.0, 2.0, 3.0, 4.0]);
        let b = NumArray32::new(vec![4.0, 3.0, 2.0, 1.0]);
        assert_eq!(a.dot(&b), 20.0);
    }

    #[test]
    fn test_dot_product_f64() {
        let a = NumArray64::new(vec![1.0, 2.0, 3.0, 4.0]);
        let b = NumArray64::new(vec![4.0, 3.0, 2.0, 1.0]);
        assert_eq!(a.dot(&b), 20.0);
    }

    #[test]
    fn test_mean_f32() {
        let a = NumArray32::new(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.mean(), 2.5);
    }

    #[test]
    fn test_mean_f64() {
        let a = NumArray64::new(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.mean(), 2.5);
    }

    #[test]
    fn test_calculate_reduced_index_1d() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let array = NumArray32::new_with_shape(data, vec![4]);
        assert_eq!(array.calculate_reduced_index(0, &[1]), 0);
        assert_eq!(array.calculate_reduced_index(1, &[1]), 0);
        assert_eq!(array.calculate_reduced_index(2, &[1]), 0);
        assert_eq!(array.calculate_reduced_index(3, &[1]), 0);
    }

    #[test]
    fn test_calculate_reduced_index_2d() {
        let shape = vec![2, 3];  // This is the shape of the data
        let reduction_shape = vec![2, 1];  // This signifies reduction along the second axis
        let num_array = NumArray32 { data: vec![], shape, strides: vec![], _ops: PhantomData }; // Strides are not used in the test
    
        // Here, we're verifying that the function calculates the reduced index correctly.
        assert_eq!(num_array.calculate_reduced_index(0, &reduction_shape), 0);
        assert_eq!(num_array.calculate_reduced_index(1, &reduction_shape), 0);
        assert_eq!(num_array.calculate_reduced_index(2, &reduction_shape), 0);
        assert_eq!(num_array.calculate_reduced_index(3, &reduction_shape), 1);
        assert_eq!(num_array.calculate_reduced_index(4, &reduction_shape), 1);
        assert_eq!(num_array.calculate_reduced_index(5, &reduction_shape), 1);
    }
    
    #[test]
    fn test_calculate_reduced_index_3d() {
        let shape = vec![2, 3, 2];  // This represents the shape of the data in 3D (e.g., 2 layers, 3 rows, 2 columns)
        let reduction_shape = vec![1, 3, 2];  // This signifies reduction along the first axis only
        let num_array = NumArray32 { data: vec![], shape, strides: vec![], _ops: PhantomData }; // Strides are not used in the test
    
        // Here, we're verifying that the function calculates the reduced index correctly.
        // Imagine the data flattened as [layer0_row0_col0, layer0_row0_col1, layer0_row1_col0, layer0_row1_col1, layer0_row2_col0, layer0_row2_col1, layer1_row0_col0, layer1_row0_col1, ...]
        // Reduction on the first axis combines layer indices, keeping other dimensions.
        assert_eq!(num_array.calculate_reduced_index(0, &reduction_shape), 0); // First element of the first layer
        assert_eq!(num_array.calculate_reduced_index(1, &reduction_shape), 1); // Second element of the first layer
        assert_eq!(num_array.calculate_reduced_index(2, &reduction_shape), 2); // First element of the second row, first layer
        assert_eq!(num_array.calculate_reduced_index(3, &reduction_shape), 3);
        assert_eq!(num_array.calculate_reduced_index(4, &reduction_shape), 4); // First element of the third row, first layer
        assert_eq!(num_array.calculate_reduced_index(5, &reduction_shape), 5);
    
        // Moving to the next layer, indices should map the same as they map to the same "reduced" position
        assert_eq!(num_array.calculate_reduced_index(6, &reduction_shape), 0); // First element of the first layer, second layer
        assert_eq!(num_array.calculate_reduced_index(7, &reduction_shape), 1);
        assert_eq!(num_array.calculate_reduced_index(8, &reduction_shape), 2);
        assert_eq!(num_array.calculate_reduced_index(9, &reduction_shape), 3);
        assert_eq!(num_array.calculate_reduced_index(10, &reduction_shape), 4);
        assert_eq!(num_array.calculate_reduced_index(11, &reduction_shape), 5);
    }

    #[test]
fn test_calculate_reduced_index_complex_3d_complex() {
    let shape = vec![3, 5, 4];  // This represents the shape of the data in 3D (3 blocks, 5 rows per block, 4 columns per row)
    let reduction_shape = vec![1, 5, 1];  // This signifies reduction along the first and third axes
    let num_array = NumArray32 { data: vec![], shape, strides: vec![], _ops: PhantomData }; // Strides are not used in this test

    // This tests a 3D array's reduction along non-adjacent axes.
    // Indices should map based on the remaining middle dimension.
    // The reduced index should vary only by changes in the middle dimension (rows), as the other two are collapsed.
    // We will test a few critical points across the dataset:
    
    // First row of the first block
    assert_eq!(num_array.calculate_reduced_index(0, &reduction_shape), 0);  // Block 0, Row 0, Column 0
    assert_eq!(num_array.calculate_reduced_index(1, &reduction_shape), 0);  // Block 0, Row 0, Column 1
    assert_eq!(num_array.calculate_reduced_index(2, &reduction_shape), 0);  // Block 0, Row 0, Column 2
    assert_eq!(num_array.calculate_reduced_index(3, &reduction_shape), 0);  // Block 0, Row 0, Column 3

    // Second row of the first block
    assert_eq!(num_array.calculate_reduced_index(4, &reduction_shape), 1);  // Block 0, Row 1, Column 0
    assert_eq!(num_array.calculate_reduced_index(7, &reduction_shape), 1);  // Block 0, Row 1, Column 3

    // Fifth row of the second block
    assert_eq!(num_array.calculate_reduced_index(39, &reduction_shape), 4); // Block 1, Row 4, Column 3

    // First row of the third block
    assert_eq!(num_array.calculate_reduced_index(40, &reduction_shape), 0); // Block 2, Row 0, Column 0
    assert_eq!(num_array.calculate_reduced_index(43, &reduction_shape), 0); // Block 2, Row 0, Column 3

    // Fifth row of the third block
    assert_eq!(num_array.calculate_reduced_index(59, &reduction_shape), 4); // Block 2, Row 4, Column 3
}

    

    #[test]
    fn test_mean_axes_1d() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let array = NumArray32::new_with_shape(data, vec![4]);
        let mean_array = array.mean_axes(&[0]);
        assert_eq!(mean_array.get_data(), &vec![2.5]);
    }

    #[test]
    fn test_mean_axes_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArray32::new_with_shape(data, vec![2, 3]);
        let mean_array = array.mean_axes(&[1]);

        assert_eq!(mean_array.shape(), &[2,1]);
        assert_eq!(mean_array.get_data(), &vec![2.0, 5.0]); // Mean along the second axis (columns)
    }

    #[test]
    fn test_mean_axes_2d_column() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArray32::new_with_shape(data, vec![2, 3]);
        // Compute mean across columns (axis 1)
        let mean_array = array.mean_axes(&[1]);
        assert_eq!(mean_array.get_data(), &vec![2.0, 5.0]); // Mean along the second axis (columns)
    }

    #[test]
    fn test_mean_axes_3d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let array = NumArray32::new_with_shape(data, vec![2, 2, 3]);
        // Compute mean across the last two axes (1 and 2)
        let mean_array = array.mean_axes(&[1, 2]);
        assert_eq!(mean_array.get_data(), &vec![3.5, 9.5]);
    }

    #[test]
    fn test_mean_axes_invalid_axis() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let array = NumArray32::new_with_shape(data, vec![4]);
        // Attempt to compute mean across an invalid axis
        let result = std::panic::catch_unwind(|| array.mean_axes(&[1]));
        assert!(result.is_err(), "Should panic due to invalid axis");
    }

}
