use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};
use std::simd::{f32x16, f64x8};

use crate::simd_ops::SimdOps;
use crate::traits::{FromU32, NumOps};

pub type NumArray32 = NumArray<f32, f32x16>;
pub type NumArray64 = NumArray<f64, f64x8>;

pub struct NumArray<T, Ops> {
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
        + FromU32,
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
        + Sum<T>,
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
    T: Clone,
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
        + Sum<T>,
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
}
