//! # Numerical Array Module
//!
//! This module provides the `NumArray` data structure for handling numerical data with SIMD operations.
//! The `NumArray` supports common mathematical operations, data manipulation, and transformation methods,
//! designed to efficiently leverage SIMD capabilities for high-performance numerical computing.
//!
//! ## Features
//!
//! - Generic numerical array implementation.
//! - Support for SIMD optimized operations.
//! - Basic arithmetic, statistical, and algebraic functions.
//! - Flexible data reshaping and manipulation.
//!
//! ## Usage
//!
//! The `NumArray` can be used directly with type aliases `NumArray32` and `NumArray64` for `f32` and `f64`
//! data types respectively, simplifying the usage in applications requiring high performance computation.
//!
//! ```rust
//! use rustynum_rs::NumArray32;
//!
//! let mut array = NumArray32::new(vec![1.0, 2.0, 3.0, 4.0]);
//! let mean_value = array.mean().item();
//! println!("Mean value: {}", mean_value);
//! ```

use std::fmt::Debug;
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};
use std::simd::{f32x16, f64x8};

use crate::num_array::linalg::matrix_multiply;
use crate::simd_ops::SimdOps;
use crate::traits::{FromU32, FromUsize, NumOps};

pub type NumArray32 = NumArray<f32, f32x16>;
pub type NumArray64 = NumArray<f64, f64x8>;

/// A generic numerical array with SIMD operations support.
///
/// # Parameters
/// * `T` - The type of elements stored in the array.
/// * `Ops` - The SIMD operations associated with type `T`.
///
/// # Example
/// ```
/// use rustynum_rs::NumArray32;
/// let array = NumArray32::new(vec![1.0, 2.0, 3.0, 4.0]);
/// ```
pub struct NumArray<T, Ops>
where
    T: Debug,
{
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    _ops: PhantomData<Ops>,
}

impl<T, Ops> NumArray<T, Ops>
where
    T: Debug, // Minimal trait bounds for basic operations
{
    /// Retrieves the shape of the array.
    ///
    /// # Returns
    /// A reference to the shape vector.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<T, Ops> NumArray<T, Ops>
where
    T: Copy + Debug, // Only require what's absolutely necessary for this operation
{
    /// Creates a new 1D array from the given data.
    ///
    /// # Parameters
    /// * `data` - The vector of elements.
    ///
    /// # Returns
    /// A new `NumArray` instance with shape inferred as 1D.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArray32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let array = NumArray32::new(data);
    /// ```
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
        + PartialOrd
        + FromU32
        + FromUsize
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
    /// Creates a new array from the given data and a specific shape.
    ///
    /// # Parameters
    /// * `data` - The vector of elements.
    /// * `shape` - A vector of dimensions defining the shape of the array.
    ///
    /// # Returns
    /// A new `NumArray` instance.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArray32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let array = NumArray32::new_with_shape(data, vec![2, 3]);
    /// ```
    pub fn new_with_shape(data: Vec<T>, shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
            _ops: PhantomData,
        }
    }

    /// Creates a new array filled with zeros.
    ///
    /// # Parameters
    /// * `shape` - A vector of dimensions defining the shape of the array.
    ///
    /// # Returns
    /// A new `NumArray` instance filled with zeros.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArray32;
    /// let zeros_array = NumArray32::zeros(vec![2, 3]);
    /// println!("Zeros array: {:?}", zeros_array.get_data());
    /// ```
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        let data = vec![T::default(); size];
        let strides = Self::compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
            _ops: PhantomData,
        }
    }

    /// Creates a new array filled with ones.
    ///
    /// # Parameters
    /// * `shape` - A vector of dimensions defining the shape of the array.
    ///
    /// # Returns
    /// A new `NumArray` instance filled with ones.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArray32;
    /// let ones_array = NumArray32::ones(vec![2, 3]);
    /// println!("Ones array: {:?}", ones_array.get_data());
    /// ```
    pub fn ones(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        let data = vec![T::from_usize(1); size];
        let strides = Self::compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
            _ops: PhantomData,
        }
    }

    /// Transposes a 2D matrix from row-major to column-major format.
    ///
    /// # Returns
    /// A new `NumArray` instance that is the transpose of the original matrix.
    pub fn transpose(&self) -> Self {
        assert!(
            self.shape.len() == 2,
            "Transpose is only valid for 2D matrices."
        );

        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut transposed_data = vec![T::default(); rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                transposed_data[j * rows + i] = self.data[i * cols + j];
            }
        }

        NumArray::new_with_shape(transposed_data, vec![cols, rows])
    }

    /// Retrieves a reference to the underlying data vector.
    ///
    /// # Returns
    /// A reference to the data vector.
    pub fn get_data(&self) -> &Vec<T> {
        return &self.data;
    }

    /// Reshapes the array to a new shape.
    ///
    /// # Parameters
    /// * `new_shape` - A vector of dimensions defining the new shape of the array.
    ///
    /// # Returns
    /// A new `NumArray` instance with the specified shape.
    ///
    /// # Panics
    /// Panics if the new shape is not compatible with the data size.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArray32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let array = NumArray32::new_with_shape(data, vec![2, 3]);
    /// let reshaped_array = array.reshape(&vec![3, 2]);
    /// println!("Reshaped array: {:?}", reshaped_array.get_data());
    /// ```
    pub fn reshape(&self, new_shape: &Vec<usize>) -> Self {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(
            new_size,
            self.data.len(),
            "New shape must be compatible with data size."
        );

        NumArray::new_with_shape(self.data.clone(), new_shape.clone())
    }

    /// Computes the strides for the given shape.
    ///
    /// # Parameters
    /// * `shape` - A reference to the shape vector.
    ///
    /// # Returns
    /// A vector of strides.
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

    /// Computes the dot product of two arrays, which can be vectors or matrices.
    ///
    /// # Parameters
    /// * `other` - A reference to the other `NumArray` instance.
    ///
    /// # Returns
    /// The dot product of the two arrays, which could be a scalar or an array depending on the inputs.
    ///
    /// # Panics
    /// Panics if the dimensions do not align for the dot product or matrix multiplication rules.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArray32;
    /// let a = NumArray32::new(vec![1.0, 2.0, 3.0, 4.0]);
    /// let b = NumArray32::new(vec![4.0, 3.0, 2.0, 1.0]);
    /// let dot_product = a.dot(&b).item();
    /// println!("Dot product: {}", dot_product);
    /// ```
    pub fn dot(&self, other: &Self) -> NumArray<T, Ops> {
        let self_shape = self.shape();
        let other_shape = other.shape();

        // Ensure the dimensions are compatible for dot product or matrix multiplication
        assert!(
            self_shape.last() == Some(&other_shape[0]),
            "Dimensions must align for dot product or matrix multiplication"
        );

        if self_shape.len() == 1 && other_shape.len() == 1 {
            // Vector dot product
            assert_eq!(
                self_shape[0], other_shape[0],
                "Vectors must be of the same length for dot product."
            );
            let dot_product = Ops::dot_product(&self.data, &other.data);
            NumArray::new(vec![dot_product])
        } else {
            // Matrix-vector or matrix-matrix multiplication
            matrix_multiply(self, other)
        }
    }

    /// Creates a new 1D array with linearly spaced values between `start` and `stop`.
    ///
    /// # Parameters
    /// * `start` - The start value of the sequence.
    /// * `stop` - The end value of the sequence.
    /// * `num` - The number of values to generate.
    ///
    /// # Returns
    /// A new `NumArray` instance with the specified linear space.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArray32;
    /// let linspace_array = NumArray32::linspace(0.0, 1.0, 5);
    /// println!("Linspace array: {:?}", linspace_array.get_data());
    /// ```
    pub fn linspace(start: T, stop: T, num: usize) -> Self {
        assert!(num > 1, "num must be greater than 1 for linspace.");
        let step = (stop - start) / T::from_usize(num - 1);
        let mut data = Vec::with_capacity(num);
        let mut current = start;
        for _ in 0..num {
            data.push(current);
            current = current + step;
        }
        let shape = vec![num];
        let strides = vec![1];
        Self {
            data,
            shape,
            strides,
            _ops: PhantomData,
        }
    }

    /// Creates a new 1D array with values starting from `start` to `stop` with a given `step`.
    ///
    /// # Parameters
    /// * `start` - The start value of the sequence.
    /// * `stop` - The end value of the sequence.
    /// * `step` - The step value between each pair of consecutive values.
    ///
    /// # Returns
    /// A new `NumArray` instance with the specified range.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArray32;
    /// let arange_array = NumArray32::arange(0.0, 1.0, 0.2);
    /// println!("Arange array: {:?}", arange_array.get_data());
    /// ```
    pub fn arange(start: T, stop: T, step: T) -> Self {
        assert!(
            step > T::default(),
            "step must be greater than 0 for arange."
        );
        let mut data = Vec::new();
        let mut current = start;
        while current < stop {
            data.push(current);
            current = current + step;
        }
        let shape = vec![data.len()];
        let strides = vec![1];
        Self {
            data,
            shape,
            strides,
            _ops: PhantomData,
        }
    }

    /// Computes the mean of the array.
    ///
    /// # Returns
    /// A new `NumArray` instance containing the mean value.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArray32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let array = NumArray32::new(data);
    /// let mean_array = array.mean();
    /// println!("Mean array: {:?}", mean_array.get_data());
    /// ```
    pub fn mean(&self) -> NumArray<T, Ops> {
        let sum: T = Ops::sum(&self.data);
        let count = T::from_u32(self.data.len() as u32);
        let mean = sum / count;
        NumArray::new(vec![mean])
    }

    /// Retrieves the single element of the array.
    ///
    /// # Returns
    /// The single element of the array.
    ///
    /// # Panics
    /// Panics if the array does not have exactly one element.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArray32;
    /// let data = vec![1.0];
    /// let array = NumArray32::new(data);
    /// let item = array.item();
    /// println!("Item: {}", item);
    /// ```
    pub fn item(&self) -> T {
        assert_eq!(self.data.len(), 1, "Array must have exactly one element.");
        self.data[0]
    }

    /// Calculates the reduced index for a given index and reduction shape.
    ///
    /// # Parameters
    /// * `index` - The original index.
    /// * `reduction_shape` - A reference to the reduction shape vector.
    ///
    /// # Returns
    /// The reduced index.
    pub fn calculate_reduced_index(&self, index: usize, reduction_shape: &[usize]) -> usize {
        let mut reduced_index = 0;
        let mut stride = 1;
        let mut accumulated_strides = 1; // To handle collapsed dimensions correctly

        for (_i, (&original_dim, &reduced_dim)) in self
            .shape
            .iter()
            .zip(reduction_shape.iter())
            .rev()
            .enumerate()
        {
            let index_in_dim = (index / stride) % original_dim;

            if reduced_dim != 1 {
                reduced_index += index_in_dim * accumulated_strides;
                accumulated_strides *= original_dim; // Update only when not collapsing
            }

            stride *= original_dim; // Always update stride to traverse dimensions
        }

        reduced_index
    }

    fn squeeze(&self) -> NumArray<T, Ops> {
        let mut new_shape = self.shape.clone();
        let mut new_data = self.data.clone();

        let mut new_shape = new_shape
            .into_iter()
            .filter(|&x| x != 1)
            .collect::<Vec<_>>();

        NumArray::new_with_shape(self.data.clone(), new_shape)
    }

    /// Computes the mean along the specified axes.
    ///
    /// # Parameters
    /// * `axes` - An optional reference to a vector of axes to compute the mean along.
    ///
    /// # Returns
    /// A new `NumArray` instance containing the mean values along the specified axes.
    ///
    /// # Panics
    /// Panics if any of the specified axes is out of bounds.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArray32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let array = NumArray32::new_with_shape(data, vec![2, 3]);
    /// let mean_array = array.mean_axes(Some(&[1]));
    /// println!("Mean array: {:?}", mean_array.get_data());
    /// ```
    pub fn mean_axes(&self, axes: Option<&[usize]>) -> NumArray<T, Ops> {
        match axes {
            Some(axes) => {
                for &axis in axes {
                    assert!(axis < self.shape.len(), "Axis {} out of bounds.", axis);
                }

                let mut reduced_shape = self.shape.clone();
                let mut total_elements_to_reduce = 1;

                for &axis in axes {
                    total_elements_to_reduce *= self.shape[axis];
                    reduced_shape[axis] = 1; // Marking this axis for reduction
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
                // let's squeeze the reduced shape
                reduced_shape = reduced_shape
                    .into_iter()
                    .filter(|&x| x != 1)
                    .collect::<Vec<_>>();
                NumArray::new_with_shape(reduced_data, reduced_shape)
            }
            None => self.mean(),
        }
    }

    /// Computes the minimum value in the array.
    ///
    /// # Returns
    /// The minimum value in the array.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArray32;
    /// let array = NumArray32::new(vec![1.0, 2.0, 3.0, 4.0]);
    /// let min_value = array.min();
    /// println!("Min value: {}", min_value);
    /// ```
    pub fn min(&self) -> T {
        Ops::min(&self.data)
    }

    /// Computes the maximum value in the array.
    ///
    /// # Returns
    /// The maximum value in the array.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArray32;
    /// let array = NumArray32::new(vec![1.0, 2.0, 3.0, 4.0]);
    /// let max_value = array.max();
    /// println!("Max value: {}", max_value);
    /// ```
    pub fn max(&self) -> T {
        Ops::max(&self.data)
    }

    /// Normalizes the array.
    ///
    /// # Returns
    /// A new `NumArray` instance with normalized data.
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

    /// Creates a slice of the array.
    ///
    /// # Parameters
    /// * `start` - The start index of the slice.
    /// * `end` - The end index of the slice.
    ///
    /// # Returns
    /// A new `NumArray` instance representing the slice.
    ///
    /// # Panics
    /// Panics if the start index exceeds the end index or if the end index exceeds the data length.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArray32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let array = NumArray32::new(data);
    /// let sliced_array = array.slice(1, 4);
    /// println!("Sliced array: {:?}", sliced_array.get_data());
    /// ```
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

    /// Extracts a slice representing a row from the matrix.
    ///
    /// # Parameters
    /// * `row` - The row index to extract.
    ///
    /// # Returns
    /// A slice representing the row from the matrix.
    ///
    /// # Panics
    /// Panics if the array is not 2D or if the row index is out of bounds.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArray32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let array = NumArray32::new_with_shape(data, vec![2, 3]);
    /// let row_slice = array.row_slice(1);
    /// println!("Row slice: {:?}", row_slice);
    /// ```
    pub fn row_slice(&self, row: usize) -> &[T] {
        let shape = self.shape();
        assert_eq!(shape.len(), 2, "Only 2D arrays are supported.");
        let start = row * shape[1];
        let end = start + shape[1];
        &self.data[start..end]
    }

    /// Extracts a slice representing a column from the matrix (more complex due to non-contiguity).
    pub fn column_slice(&self, col: usize) -> Vec<T> {
        assert_eq!(self.shape().len(), 2, "Only 2D arrays are supported.");
        (0..self.shape()[0])
            .map(|i| self.data[i * self.shape()[1] + col])
            .collect()
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
    /// Constructs a new `NumArray` from the given data.
    ///
    /// The shape of the array is assumed to be 1-dimensional based on the length of the data.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to initialize the array with.
    ///
    /// # Returns
    ///
    /// A new `NumArray` with the given data.
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
    T: Clone + Debug,
{
    /// Returns a new `NumArray` that is a clone of the current instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustynum_rs::NumArray32;
    ///
    /// let arr1 = NumArray32::new(vec![1.0, 2.0, 3.0]);
    /// let arr2 = arr1.clone();
    /// ```
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
    /// Creates a new `NumArray` from a slice of data.
    ///
    /// # Arguments
    ///
    /// * `data` - The slice of data to create the `NumArray` from.
    ///
    /// # Returns
    ///
    /// A new `NumArray` with the provided data.
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

impl<T, Ops> NumArray<T, Ops>
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
    /// Retrieves an element at a given multidimensional index.
    ///
    /// # Parameters
    /// * `indices` - A slice representing the multidimensional index.
    ///
    /// # Returns
    /// The element at the given index.
    ///
    /// # Panics
    /// Panics if the indices are out of bounds.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArray32;
    /// let array = NumArray32::new(vec![1.0, 2.0, 3.0, 4.0]);
    /// let element = array.get(&[1]);
    /// ```
    pub fn get(&self, indices: &[usize]) -> T {
        assert!(
            indices.len() == self.shape.len(),
            "Indices must match the dimensions of the array."
        );
        let index = self.calculate_linear_index(indices);
        self.data[index]
    }

    /// Sets an element at a given multidimensional index.
    ///
    /// # Parameters
    /// * `indices` - A slice representing the multidimensional index.
    /// * `value` - The value to set at the given index.
    ///
    /// # Panics
    /// Panics if the indices are out of bounds.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArray32;
    /// let mut array = NumArray32::new(vec![1.0, 2.0, 3.0, 4.0]);
    /// array.set(&[1], 5.0);
    /// ```
    pub fn set(&mut self, indices: &[usize], value: T) {
        assert!(
            indices.len() == self.shape.len(),
            "Indices must match the dimensions of the array."
        );
        let index = self.calculate_linear_index(indices);
        self.data[index] = value;
    }

    /// Calculates a linear index from a multidimensional index based on the strides of the array.
    ///
    /// # Parameters
    /// * `indices` - A slice of usizes representing the indices for each dimension.
    ///
    /// # Returns
    /// The linear index corresponding to the multidimensional index.
    fn calculate_linear_index(&self, indices: &[usize]) -> usize {
        indices
            .iter()
            .zip(self.strides.iter())
            .fold(0, |acc, (&idx, &stride)| acc + idx * stride)
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
    fn test_zeros_array() {
        let shape = vec![2, 3]; // 2x3 matrix
        let zeros_array = NumArray32::zeros(shape.clone());
        assert_eq!(zeros_array.shape(), shape.as_slice());
        assert_eq!(zeros_array.get_data(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ones_array() {
        let shape = vec![2, 3]; // 2x3 matrix
        let ones_array = NumArray32::ones(shape.clone());
        assert_eq!(ones_array.shape(), shape.as_slice());
        assert_eq!(ones_array.get_data(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_matrix_transpose() {
        let matrix = NumArray32::new_with_shape(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        );

        let transposed = matrix.transpose();

        let expected_data = vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];

        assert_eq!(transposed.shape(), &[3, 3]);
        assert_eq!(transposed.get_data(), &expected_data);
    }

    #[test]
    fn test_non_square_matrix_transpose() {
        let matrix =
            NumArray32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);

        let transposed = matrix.transpose();

        let expected_data = vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0];

        assert_eq!(transposed.shape(), &[4, 2]);
        assert_eq!(transposed.get_data(), &expected_data);
    }

    #[test]
    fn test_reshape_successfully() {
        let array = NumArray32::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let new_shape = vec![2, 3]; // New shape compatible with the size of data (2*3 = 6)
        let reshaped_array = array.reshape(&new_shape.clone());
        assert_eq!(reshaped_array.shape(), new_shape.as_slice());
    }

    #[test]
    fn test_reshape_and_strides() {
        let array = NumArray32::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let new_shape = vec![2, 4]; // Reshape to a 2x4 matrix
        let reshaped_array = array.reshape(&new_shape.clone());
        assert_eq!(reshaped_array.shape(), new_shape.as_slice());
        // Check strides for a 2x4 matrix
        let expected_strides = vec![4, 1]; // Moving to the next row jumps 4 elements, column jump is 1
        assert_eq!(reshaped_array.strides, expected_strides);
    }

    #[test]
    fn test_dot_product_f32() {
        let a = NumArray32::new(vec![1.0, 2.0, 3.0, 4.0]);
        let b = NumArray32::new(vec![4.0, 3.0, 2.0, 1.0]);
        assert_eq!(a.dot(&b).get_data(), &[20.0]);
    }

    #[test]
    fn test_dot_product_f64() {
        let a = NumArray64::new(vec![1.0, 2.0, 3.0, 4.0]);
        let b = NumArray64::new(vec![4.0, 3.0, 2.0, 1.0]);
        assert_eq!(a.dot(&b).get_data(), &[20.0]);
    }

    #[test]
    #[should_panic(expected = "Dimensions must align for dot product or matrix multiplication")]
    fn test_vector_dot_product_dimension_mismatch() {
        let a = NumArray32::new(vec![1.0, 2.0, 3.0]);
        let b = NumArray32::new(vec![1.0, 2.0]);
        a.dot(&b);
    }

    #[test]
    fn test_matrix_vector_multiply_correct() {
        let matrix = NumArray32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let vector = NumArray32::new_with_shape(vec![1.0, 2.0, 3.0], vec![3]);
        let result = matrix.dot(&vector);
        assert_eq!(result.get_data(), &[14.0, 32.0]);
    }

    #[test]
    fn test_matrix_matrix_multiply_correct() {
        let a = NumArray32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = NumArray32::new_with_shape(vec![2.0, 0.0, 1.0, 3.0], vec![2, 2]);
        let result = a.dot(&b);
        // Expected result of multiplication:
        // [1*2 + 2*1, 1*0 + 2*3]
        // [3*2 + 4*1, 3*0 + 4*3]
        // [2 + 2, 0 + 6]
        // [6 + 4, 0 + 12]
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.get_data(), &[4.0, 6.0, 10.0, 12.0]);
    }

    #[test]
    fn test_arange() {
        let arange_array = NumArray32::arange(0.0, 1.0, 0.2);
        assert_eq!(arange_array.get_data(), &[0.0, 0.2, 0.4, 0.6, 0.8]);
    }

    #[test]
    fn test_linspace() {
        let linspace_array = NumArray32::linspace(0.0, 1.0, 5);
        assert_eq!(linspace_array.get_data(), &[0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn test_mean_f32() {
        let a = NumArray32::new(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.mean().item(), 2.5);
    }

    #[test]
    fn test_mean_f64() {
        let a = NumArray64::new(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.mean().item(), 2.5);
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
        let shape = vec![2, 3]; // This is the shape of the data
        let reduction_shape = vec![2, 1]; // This signifies reduction along the second axis
        let num_array = NumArray32 {
            data: vec![],
            shape,
            strides: vec![],
            _ops: PhantomData,
        }; // Strides are not used in the test

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
        let shape = vec![2, 3, 2]; // This represents the shape of the data in 3D (e.g., 2 layers, 3 rows, 2 columns)
        let reduction_shape = vec![1, 3, 2]; // This signifies reduction along the first axis only
        let num_array = NumArray32 {
            data: vec![],
            shape,
            strides: vec![],
            _ops: PhantomData,
        }; // Strides are not used in the test

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
        let shape = vec![3, 5, 4]; // This represents the shape of the data in 3D (3 blocks, 5 rows per block, 4 columns per row)
        let reduction_shape = vec![1, 5, 1]; // This signifies reduction along the first and third axes
        let num_array = NumArray32 {
            data: vec![],
            shape,
            strides: vec![],
            _ops: PhantomData,
        }; // Strides are not used in this test

        // This tests a 3D array's reduction along non-adjacent axes.
        // Indices should map based on the remaining middle dimension.
        // The reduced index should vary only by changes in the middle dimension (rows), as the other two are collapsed.
        // We will test a few critical points across the dataset:

        // First row of the first block
        assert_eq!(num_array.calculate_reduced_index(0, &reduction_shape), 0); // Block 0, Row 0, Column 0
        assert_eq!(num_array.calculate_reduced_index(1, &reduction_shape), 0); // Block 0, Row 0, Column 1
        assert_eq!(num_array.calculate_reduced_index(2, &reduction_shape), 0); // Block 0, Row 0, Column 2
        assert_eq!(num_array.calculate_reduced_index(3, &reduction_shape), 0); // Block 0, Row 0, Column 3

        // Second row of the first block
        assert_eq!(num_array.calculate_reduced_index(4, &reduction_shape), 1); // Block 0, Row 1, Column 0
        assert_eq!(num_array.calculate_reduced_index(7, &reduction_shape), 1); // Block 0, Row 1, Column 3

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
        let mean_array = array.mean_axes(Some(&[0]));
        assert_eq!(mean_array.get_data(), &vec![2.5]);
    }

    #[test]
    fn test_mean_axes_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArray32::new_with_shape(data, vec![2, 3]);
        let mean_array = array.mean_axes(Some(&[1]));

        assert_eq!(mean_array.shape(), &[2]);
        assert_eq!(mean_array.get_data(), &vec![2.0, 5.0]); // Mean along the second axis (columns)
    }

    #[test]
    fn test_mean_axes_2d_column() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArray32::new_with_shape(data, vec![2, 3]);
        // Compute mean across columns (axis 1)
        let mean_array = array.mean_axes(Some(&[1]));
        assert_eq!(mean_array.get_data(), &vec![2.0, 5.0]); // Mean along the second axis (columns)
    }

    #[test]
    fn test_mean_axes_3d() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let array = NumArray32::new_with_shape(data, vec![2, 2, 3]);
        // Compute mean across the last two axes (1 and 2)
        let mean_array = array.mean_axes(Some(&[1, 2]));
        assert_eq!(mean_array.get_data(), &vec![3.5, 9.5]);
    }

    #[test]
    fn test_mean_axes_invalid_axis() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let array = NumArray32::new_with_shape(data, vec![4]);
        // Attempt to compute mean across an invalid axis
        let result = std::panic::catch_unwind(|| array.mean_axes(Some(&[1])));
        assert!(result.is_err(), "Should panic due to invalid axis");
    }

    #[test]
    fn test_get_and_set_single_element() {
        let mut array = NumArray32::new_with_shape(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        // Test getting an element
        assert_eq!(array.get(&[0, 1]), 20.0);

        // Test setting an element
        array.set(&[0, 1], 25.0);
        assert_eq!(array.get(&[0, 1]), 25.0);
    }

    #[test]
    fn test_get_and_set_edge_cases() {
        let mut array = NumArray32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        // Get and set first element
        assert_eq!(array.get(&[0, 0]), 1.0);
        array.set(&[0, 0], 10.0);
        assert_eq!(array.get(&[0, 0]), 10.0);

        // Get and set last element
        assert_eq!(array.get(&[1, 2]), 6.0);
        array.set(&[1, 2], 60.0);
        assert_eq!(array.get(&[1, 2]), 60.0);
    }

    #[test]
    #[should_panic(expected = "index out of bounds: the len is 4 but the index is 6")]
    fn test_get_out_of_bounds() {
        let array = NumArray32::new_with_shape(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        // Attempt to access an out-of-bounds index
        let _ = array.get(&[2, 2]);
    }

    #[test]
    #[should_panic(expected = "index out of bounds: the len is 4 but the index is 6")]
    fn test_set_out_of_bounds() {
        let mut array = NumArray32::new_with_shape(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        // Attempt to set an out-of-bounds index
        array.set(&[2, 2], 50.0);
    }

    #[test]
    #[should_panic(expected = "Indices must match the dimensions of the array.")]
    fn test_get_wrong_dimension_count() {
        let array = NumArray32::new_with_shape(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        // Attempt to access with incorrect number of dimensions
        let _ = array.get(&[1]);
    }

    #[test]
    #[should_panic(expected = "Indices must match the dimensions of the array.")]
    fn test_set_wrong_dimension_count() {
        let mut array = NumArray32::new_with_shape(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        // Attempt to set with incorrect number of dimensions
        array.set(&[1], 50.0);
    }

    #[test]
    fn test_row_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArray32::new_with_shape(data, vec![2, 3]);
        let row_slice = array.row_slice(1);
        assert_eq!(row_slice, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_column_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArray32::new_with_shape(data, vec![2, 3]);
        let column_slice = array.column_slice(1);
        assert_eq!(column_slice, vec![2.0, 5.0]);
    }
}
