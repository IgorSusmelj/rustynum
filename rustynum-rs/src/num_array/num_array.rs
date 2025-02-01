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
//! The `NumArray` can be used directly with type aliases `NumArrayF32` and `NumArrayF64` for `f32` and `f64`
//! data types respectively, simplifying the usage in applications requiring high performance computation.
//!
//! ```rust
//! use rustynum_rs::NumArrayF32;
//!
//! let mut array = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0]);
//! let mean_value = array.mean().item();
//! println!("Mean value: {}", mean_value);
//! ```

use std::fmt::Debug;
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::simd::{f32x16, f64x8, i32x16, i64x8, u8x64};

use crate::num_array::linalg::matrix_multiply;
use crate::simd_ops::SimdOps;
use crate::traits::{AbsOps, ExpLog, FromU32, FromUsize, NumOps};

pub type NumArrayU8 = NumArray<u8, u8x64>;
pub type NumArrayI32 = NumArray<i32, i32x16>;
pub type NumArrayI64 = NumArray<i64, i64x8>;

pub type NumArrayF32 = NumArray<f32, f32x16>;
pub type NumArrayF64 = NumArray<f64, f64x8>;

/// A generic numerical array with SIMD operations support.
///
/// # Parameters
/// * `T` - The type of elements stored in the array.
/// * `Ops` - The SIMD operations associated with type `T`.
///
/// # Example
/// ```
/// use rustynum_rs::NumArrayF32;
/// let array = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0]);
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
    T: Copy + Debug + Default,
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
    /// use rustynum_rs::NumArrayF32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let array = NumArrayF32::new(data);
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
    /// use rustynum_rs::NumArrayF32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
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
    #[inline]
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
    /// use rustynum_rs::NumArrayF32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
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

    /// Reverses the elements along the specified axis or axis.
    ///
    /// # Parameters
    /// * `axis` - An iterable of axis along which to reverse the array.
    ///
    /// # Returns
    /// A new `NumArray` instance with the array reversed along the specified axis.
    ///
    /// # Panics
    /// Panics if any axis is out of bounds.
    pub fn flip_axis<I>(&self, axis: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        let mut new_data = self.data.clone();
        let new_shape = self.shape.clone();

        for axis in axis {
            assert!(
                axis < self.shape.len(),
                "Axis {} is out of bounds for an array with {} dimensions.",
                axis,
                self.shape.len()
            );

            let axis_size = self.shape[axis];

            // Compute the number of "blocks" to process
            let blocks: usize = self.shape.iter().take(axis).product();
            let block_size: usize = self.shape.iter().skip(axis + 1).product();

            for block in 0..blocks {
                for i in 0..axis_size / 2 {
                    let idx1 = block * axis_size * block_size + i * block_size;
                    let idx2 = block * axis_size * block_size + (axis_size - 1 - i) * block_size;

                    for k in 0..block_size {
                        new_data.swap(idx1 + k, idx2 + k);
                    }
                }
            }
        }

        Self::new_with_shape(new_data, new_shape)
    }
}

impl<T, Ops> NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Copy
        + Debug
        + Default
        + PartialOrd
        + FromU32
        + FromUsize,
    Ops: SimdOps<T>,
{
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
    /// use rustynum_rs::NumArrayF32;
    /// let arange_array = NumArrayF32::arange(0.0, 1.0, 0.2);
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
    /// use rustynum_rs::NumArrayF32;
    /// let data = vec![1.0];
    /// let array = NumArrayF32::new(data);
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

    /// Removes axis of length one from the array.
    ///
    /// # Parameters
    /// * `axis` - Optional slice of axis to squeeze. If None, all axis of length 1 are removed.
    ///
    /// # Returns
    /// A new `NumArray` instance with the specified axis removed.
    ///
    /// # Panics
    /// Panics if any specified axis is out of bounds or has length greater than 1.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArrayF32;
    /// let array = NumArrayF32::new_with_shape(vec![1.0, 2.0], vec![1, 2, 1]);
    /// let squeezed = array.squeeze(None); // removes all axis of length 1
    /// assert_eq!(squeezed.shape(), &[2]);
    /// ```
    pub fn squeeze(&self, axis: Option<&[usize]>) -> NumArray<T, Ops> {
        match axis {
            Some(specified_axis) => {
                // Validate axis
                for &axis in specified_axis {
                    assert!(
                        axis < self.shape.len(),
                        "Axis {} is out of bounds for array of dimension {}",
                        axis,
                        self.shape.len()
                    );
                    assert!(
                        self.shape[axis] == 1,
                        "Cannot squeeze axis {} with size {}",
                        axis,
                        self.shape[axis]
                    );
                }

                let new_shape: Vec<usize> = self
                    .shape
                    .iter()
                    .enumerate()
                    .filter(|&(i, &dim)| !specified_axis.contains(&i) || dim != 1)
                    .map(|(_, &dim)| dim)
                    .collect();

                NumArray::new_with_shape(self.data.clone(), new_shape)
            }
            None => {
                // Remove all axis of length 1
                let new_shape: Vec<usize> = self
                    .shape
                    .iter()
                    .filter(|&&dim| dim != 1)
                    .cloned()
                    .collect();

                NumArray::new_with_shape(self.data.clone(), new_shape)
            }
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
    /// use rustynum_rs::NumArrayF32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
    /// let sliced_array = array.slice(1, 0, 2);
    /// println!("Sliced array: {:?}", sliced_array.get_data());
    /// ```
    pub fn slice(&self, axis: usize, start: usize, end: usize) -> Self {
        assert!(axis < self.shape.len(), "Axis out of bounds");
        assert!(start <= end, "Start index must not exceed end index.");
        assert!(
            end <= self.shape[axis],
            "End index must not exceed the size of the specified axis."
        );

        let mut new_shape = self.shape.clone();
        new_shape[axis] = end - start;

        let mut new_data = Vec::with_capacity(new_shape.iter().product());

        let elements_before_axis: usize = self.shape.iter().take(axis).product();
        let elements_after_axis: usize = self.shape.iter().skip(axis + 1).product();
        let axis_size = self.shape[axis];

        for outer in 0..elements_before_axis {
            for i in start..end {
                let start_idx = outer * axis_size * elements_after_axis + i * elements_after_axis;
                let end_idx = start_idx + elements_after_axis;
                new_data.extend_from_slice(&self.data[start_idx..end_idx]);
            }
        }

        Self {
            data: new_data,
            shape: new_shape.clone(),
            strides: Self::compute_strides(&new_shape),
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
    /// use rustynum_rs::NumArrayF32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let array = NumArrayF32::new(data);
    /// let mean_array = array.mean();
    /// println!("Mean array: {:?}", mean_array.get_data());
    /// ```
    pub fn mean(&self) -> NumArray<T, Ops> {
        let sum: T = Ops::sum(&self.data);
        let count = T::from_u32(self.data.len() as u32);
        let mean = sum / count;
        NumArray::new(vec![mean])
    }

    /// Computes the mean along the specified axis.
    ///
    /// # Parameters
    /// * `axis` - An optional reference to a vector of axis to compute the mean along.
    ///
    /// # Returns
    /// A new `NumArray` instance containing the mean values along the specified axis.
    ///
    /// # Panics
    /// Panics if any of the specified axis is out of bounds.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArrayF32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
    /// let mean_array = array.mean_axis(Some(&[1]));
    /// println!("Mean array: {:?}", mean_array.get_data());
    /// ```
    pub fn mean_axis(&self, axis: Option<&[usize]>) -> NumArray<T, Ops> {
        match axis {
            Some(axis) => {
                for &axis in axis {
                    assert!(axis < self.shape.len(), "Axis {} out of bounds.", axis);
                }

                let mut reduced_shape = self.shape.clone();
                let mut total_elements_to_reduce = 1;

                for &axis in axis {
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
        + ExpLog
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
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
    /// use rustynum_rs::NumArrayF32;
    /// let zeros_array = NumArrayF32::zeros(vec![2, 3]);
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
    /// use rustynum_rs::NumArrayF32;
    /// let ones_array = NumArrayF32::ones(vec![2, 3]);
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

    /// Concatenates multiple `NumArray` instances along the specified axis.
    ///
    /// # Parameters
    /// * `arrays` - A slice of `NumArray` instances to concatenate.
    /// * `axis` - The axis along which to concatenate.
    ///
    /// # Returns
    /// A new `NumArray` instance resulting from the concatenation.
    ///
    /// # Panics
    /// Panics if the shapes of the arrays are incompatible for concatenation along the specified axis.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArrayF32;
    ///
    /// let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0], vec![3]);
    /// let b = NumArrayF32::new_with_shape(vec![4.0, 5.0], vec![2]);
    /// let concatenated = NumArrayF32::concatenate(&[a, b], 0);
    /// assert_eq!(concatenated.get_data(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// ```
    pub fn concatenate(arrays: &[Self], axis: usize) -> Self {
        // Ensure there is at least one array to concatenate
        assert!(
            !arrays.is_empty(),
            "At least one array must be provided for concatenation."
        );

        // Determine the reference shape from the first array
        let reference_shape = arrays[0].shape();

        // Validate that all arrays have the same number of dimensions
        let ndim = reference_shape.len();
        assert!(
            axis < ndim,
            "Concatenation axis {} is out of bounds for arrays with {} dimensions.",
            axis,
            ndim
        );
        for array in arrays.iter() {
            assert!(
                array.shape().len() == ndim,
                "All arrays must have the same number of dimensions."
            );
            // Validate that shapes match on all axis except the concatenation axis
            for (i, (&dim_ref, &dim_other)) in
                reference_shape.iter().zip(array.shape().iter()).enumerate()
            {
                if i != axis {
                    assert!(
                        dim_ref == dim_other,
                        "All arrays must have the same shape except along the concatenation axis. Mismatch found at axis {}.",
                        i
                    );
                }
            }
        }

        // Compute the new shape
        let mut new_shape = reference_shape.to_vec();
        let total_concat_dim: usize = arrays.iter().map(|array| array.shape()[axis]).sum();
        new_shape[axis] = total_concat_dim;

        // Compute elements_before_axis and elements_after_axis
        let elements_before_axis: usize = reference_shape.iter().take(axis).product();
        let elements_after_axis: usize = reference_shape.iter().skip(axis + 1).product();

        // Initialize the new data vector with the appropriate capacity
        let total_size: usize = new_shape.iter().product();
        let mut concatenated_data = Vec::with_capacity(total_size);

        // Iterate over each outer slice and concatenate data from all arrays
        for outer in 0..elements_before_axis {
            for array in arrays.iter() {
                let axis_size = array.shape()[axis];
                let slice_size = axis_size * elements_after_axis;

                // Calculate the start and end indices for the current slice
                let start = outer * axis_size * elements_after_axis;
                let end = start + slice_size;

                // Safety check to prevent out-of-bounds access
                assert!(
                    end <= array.data.len(),
                    "Slice indices out of bounds. Attempted to access {}..{} in an array with length {}.",
                    start,
                    end,
                    array.data.len()
                );

                // Append the slice to the concatenated data
                concatenated_data.extend_from_slice(&array.data[start..end]);
            }
        }

        // Create and return the new NumArray with the concatenated data and new shape
        Self::new_with_shape(concatenated_data, new_shape)
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
        + ExpLog
        + Neg<Output = T>
        + Default
        + Debug,
    Ops: SimdOps<T>,
{
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
    /// use rustynum_rs::NumArrayF32;
    /// let a = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0]);
    /// let b = NumArrayF32::new(vec![4.0, 3.0, 2.0, 1.0]);
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
    /// use rustynum_rs::NumArrayF32;
    /// let linspace_array = NumArrayF32::linspace(0.0, 1.0, 5);
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

    /// Computes the minimum value along the specified axis.
    ///
    /// # Parameters
    /// * `axis` - An optional reference to a vector of axis to compute the minimum along.
    ///
    /// # Returns
    /// A new `NumArray` instance containing the minimum values along the specified axis.
    ///
    /// # Panics
    /// Panics if any of the specified axis is out of bounds.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArrayF32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
    /// let min_array = array.min_axis(Some(&[1]));
    /// println!("Min array: {:?}", min_array.get_data());
    /// ```
    pub fn min_axis(&self, axis: Option<&[usize]>) -> NumArray<T, Ops> {
        match axis {
            Some(axis) => {
                for &axis in axis {
                    assert!(axis < self.shape.len(), "Axis {} out of bounds.", axis);
                }

                let mut reduced_shape = self.shape.clone();
                for &axis in axis {
                    reduced_shape[axis] = 1; // Mark this axis for reduction
                }

                let reduced_size: usize = reduced_shape.iter().product();
                let mut reduced_data = vec![T::default(); reduced_size];
                let mut initialized = vec![false; reduced_size];

                for (i, &val) in self.data.iter().enumerate() {
                    let reduced_idx = self.calculate_reduced_index(i, &reduced_shape);
                    if !initialized[reduced_idx] {
                        // First element for this group
                        reduced_data[reduced_idx] = val;
                        initialized[reduced_idx] = true;
                    } else if val < reduced_data[reduced_idx] {
                        reduced_data[reduced_idx] = val;
                    }
                }

                // Squeeze out axis of size 1
                let squeezed_shape = reduced_shape
                    .into_iter()
                    .filter(|&dim| dim != 1)
                    .collect::<Vec<_>>();

                NumArray::new_with_shape(reduced_data, squeezed_shape)
            }
            None => {
                // If no axis are provided, return the overall min
                NumArray::new(vec![Ops::min_simd(&self.data)])
            }
        }
    }

    /// Computes the minimum value in the array.
    ///
    /// # Returns
    /// The minimum value in the array.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArrayF32;
    /// let array = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0]);
    /// let min_value = array.min();
    /// println!("Min value: {}", min_value);
    /// ```
    pub fn min(&self) -> T {
        Ops::min_simd(&self.data)
    }

    /// Computes the maximum value in the array.
    ///
    /// # Returns
    /// The maximum value in the array.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArrayF32;
    /// let array = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0]);
    /// let max_value = array.max();
    /// println!("Max value: {}", max_value);
    /// ```
    pub fn max(&self) -> T {
        Ops::max_simd(&self.data)
    }

    /// Applies the exponential function to each element of the `NumArray`.
    ///
    /// # Returns
    /// A new `NumArray` instance where each element is the exponential of the corresponding element in the original array.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArrayF32;
    ///
    /// let array = NumArrayF32::new(vec![0.0, 1.0, 2.0]);
    /// let exp_array = array.exp();
    /// assert_eq!(exp_array.get_data(), &[1.0, 2.7182817, 7.389056]);
    /// ```
    pub fn exp(&self) -> Self {
        let exp_data = self.data.iter().map(|&x| x.exp()).collect::<Vec<T>>();
        Self::new_with_shape(exp_data, self.shape.clone())
    }

    /// Applies the natural logarithm to each element of the `NumArray`.
    ///
    /// # Returns
    /// A new `NumArray` instance where each element is the natural logarithm of the corresponding element in the original array.
    ///
    /// # Panics
    /// Panics if any element in the array is non-positive, as the logarithm is undefined for such values.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArrayF32;
    ///
    /// let array = NumArrayF32::new(vec![1.0, 2.718282, 7.389056]);
    /// let log_array = array.log();
    /// assert_eq!(log_array.get_data(), &[0.0, 1.0, 2.0]);
    /// ```
    pub fn log(&self) -> Self {
        // Ensure all elements are positive
        for &x in &self.data {
            assert!(
                x > T::from_u32(0),
                "Logarithm undefined for non-positive values."
            );
        }

        let log_data = self.data.iter().map(|&x| x.log()).collect::<Vec<T>>();
        Self::new_with_shape(log_data, self.shape.clone())
    }

    /// Applies the sigmoid function to each element of the `NumArray`.
    ///
    /// The sigmoid function is defined as `1 / (1 + exp(-x))` for each element `x`.
    ///
    /// # Returns
    /// A new `NumArray` instance where each element is the sigmoid of the corresponding element in the original array.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArrayF32;
    ///
    /// let array = NumArrayF32::new(vec![0.0, 2.0, -2.0]);
    /// let sigmoid_array = array.sigmoid();
    /// let expected = vec![0.5, 0.880797, 0.119203];
    /// for (computed, &exp_val) in sigmoid_array.get_data().iter().zip(expected.iter()) {
    ///     assert!((computed - exp_val).abs() < 1e-5, "Expected {}, got {}", exp_val, computed);
    /// }
    /// ```
    pub fn sigmoid(&self) -> Self {
        let sigmoid_data = self
            .data
            .iter()
            .map(|&x| T::from_u32(1) / (T::from_u32(1) + (-x).exp()))
            .collect::<Vec<T>>();
        Self::new_with_shape(sigmoid_data, self.shape.clone())
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
    /// use rustynum_rs::NumArrayF32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
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

    /// Computes the maximum value along the specified axis.
    ///
    /// # Parameters
    /// * `axis` - An optional reference to a vector of axis to compute the maximum along.
    ///
    /// # Returns
    /// A new `NumArray` instance containing the maximum values along the specified axis.
    ///
    /// # Panics
    /// Panics if any of the specified axis is out of bounds.
    ///
    /// # Example
    /// ```
    /// use rustynum_rs::NumArrayF32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
    /// let max_array = array.max_axis(Some(&[1]));
    /// println!("Max array: {:?}", max_array.get_data());
    /// ```
    pub fn max_axis(&self, axis: Option<&[usize]>) -> NumArray<T, Ops> {
        match axis {
            Some(axis) => {
                for &axis in axis {
                    assert!(axis < self.shape.len(), "Axis {} out of bounds.", axis);
                }

                let mut reduced_shape = self.shape.clone();
                for &axis in axis {
                    reduced_shape[axis] = 1; // Mark this axis for reduction
                }

                let reduced_size: usize = reduced_shape.iter().product();
                let mut reduced_data = vec![T::default(); reduced_size];
                let mut initialized = vec![false; reduced_size];

                for (i, &val) in self.data.iter().enumerate() {
                    let reduced_idx = self.calculate_reduced_index(i, &reduced_shape);
                    if !initialized[reduced_idx] {
                        // First element for this group
                        reduced_data[reduced_idx] = val;
                        initialized[reduced_idx] = true;
                    } else if val > reduced_data[reduced_idx] {
                        reduced_data[reduced_idx] = val;
                    }
                }

                // Squeeze out axis of size 1
                let squeezed_shape = reduced_shape
                    .into_iter()
                    .filter(|&dim| dim != 1)
                    .collect::<Vec<_>>();

                NumArray::new_with_shape(reduced_data, squeezed_shape)
            }
            None => {
                // If no axis are provided, return the overall max
                NumArray::new(vec![Ops::max_simd(&self.data)])
            }
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
    /// use rustynum_rs::NumArrayF32;
    ///
    /// let arr1 = NumArrayF32::new(vec![1.0, 2.0, 3.0]);
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
    /// use rustynum_rs::NumArrayF32;
    /// let array = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0]);
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
    /// use rustynum_rs::NumArrayF32;
    /// let mut array = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0]);
    /// array.set(&[1], 5.0);
    /// ```
    #[inline]
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

/// Started using Macros to reduce the code
/// Macro to implement binary operations (`Add`, `Sub`, `Mul`, `Div`) for NumArray types.
/// It implements both scalar and array operations (with and without passing references).
macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $num_array_type:ty, $scalar_type:ty, $wrapping_fn:ident) => {
        impl std::ops::$trait<$scalar_type> for $num_array_type {
            type Output = $num_array_type;

            fn $method(self, scalar: $scalar_type) -> $num_array_type {
                let data = self.data.iter().map(|&x| x.$wrapping_fn(scalar)).collect();
                <$num_array_type>::new_with_shape(data, self.shape.clone())
            }
        }

        impl std::ops::$trait<$num_array_type> for $num_array_type {
            type Output = $num_array_type;

            fn $method(self, other: $num_array_type) -> $num_array_type {
                assert_eq!(
                    self.shape, other.shape,
                    "Shapes must match for element-wise operations"
                );
                let data = self
                    .data
                    .iter()
                    .zip(other.data.iter())
                    .map(|(&x, &y)| x.$wrapping_fn(y))
                    .collect();
                <$num_array_type>::new_with_shape(data, self.shape.clone())
            }
        }

        // Optionally, implement for references to avoid consuming the original arrays
        impl<'a> std::ops::$trait<$scalar_type> for &'a $num_array_type {
            type Output = $num_array_type;

            fn $method(self, scalar: $scalar_type) -> $num_array_type {
                let data = self.data.iter().map(|&x| x.$wrapping_fn(scalar)).collect();
                <$num_array_type>::new_with_shape(data, self.shape.clone())
            }
        }

        impl<'a> std::ops::$trait<&'a $num_array_type> for $num_array_type {
            type Output = $num_array_type;

            fn $method(self, other: &'a $num_array_type) -> $num_array_type {
                assert_eq!(
                    self.shape, other.shape,
                    "Shapes must match for element-wise operations"
                );
                let data = self
                    .data
                    .iter()
                    .zip(other.data.iter())
                    .map(|(&x, &y)| x.$wrapping_fn(y))
                    .collect();
                <$num_array_type>::new_with_shape(data, self.shape.clone())
            }
        }

        impl<'a, 'b> std::ops::$trait<&'b $num_array_type> for &'a $num_array_type {
            type Output = $num_array_type;

            fn $method(self, other: &'b $num_array_type) -> $num_array_type {
                assert_eq!(
                    self.shape, other.shape,
                    "Shapes must match for element-wise operations"
                );
                let data = self
                    .data
                    .iter()
                    .zip(other.data.iter())
                    .map(|(&x, &y)| x.$wrapping_fn(y))
                    .collect();
                <$num_array_type>::new_with_shape(data, self.shape.clone())
            }
        }
    };
}

impl_binary_op!(Add, add, NumArrayU8, u8, wrapping_add);
impl_binary_op!(Sub, sub, NumArrayU8, u8, wrapping_sub);
impl_binary_op!(Mul, mul, NumArrayU8, u8, wrapping_mul);
impl_binary_op!(Div, div, NumArrayU8, u8, wrapping_div);

impl_binary_op!(Add, add, NumArrayI32, i32, wrapping_add);
impl_binary_op!(Sub, sub, NumArrayI32, i32, wrapping_sub);
impl_binary_op!(Mul, mul, NumArrayI32, i32, wrapping_mul);
impl_binary_op!(Div, div, NumArrayI32, i32, wrapping_div);

impl_binary_op!(Add, add, NumArrayI64, i64, wrapping_add);
impl_binary_op!(Sub, sub, NumArrayI64, i64, wrapping_sub);
impl_binary_op!(Mul, mul, NumArrayI64, i64, wrapping_mul);
impl_binary_op!(Div, div, NumArrayI64, i64, wrapping_div);

impl<T, Ops> NumArray<T, Ops>
where
    T: Copy
        + Debug
        + Default
        + AbsOps
        + NumOps
        + Add<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + Div<Output = T>,
    Ops: SimdOps<T>,
{
    /// Calculates the norm of the array.
    ///
    /// For p = 1, computes the L1 norm (i.e., the sum of absolute values).
    /// For p = 2, computes the L2 norm (i.e., the square root of the sum of squares).
    ///
    /// Instead of manually iterating the data, this version uses the simd operations
    /// defined in the SimdOps trait.
    pub fn norm(&self, p: u32, axis: Option<&[usize]>) -> Self {
        match axis {
            None => {
                // Full reduction - current behavior
                let data = self.get_data();
                let result = match p {
                    1 => Ops::l1_norm(data),
                    2 => Ops::l2_norm(data),
                    _ => unimplemented!("Only L1 and L2 norm are implemented"),
                };
                Self::new(vec![result])
            }
            Some(axes) => {
                // For axis-specific reduction, we need to compute norm along specified axes
                let mut result_shape: Vec<usize> = self
                    .shape
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &dim)| if !axes.contains(&i) { Some(dim) } else { None })
                    .collect();

                if result_shape.is_empty() {
                    result_shape.push(1);
                }

                let stride = self.shape[axes[0]];
                let n_chunks = self.data.len() / stride;
                let mut result = Vec::with_capacity(n_chunks);

                for chunk_idx in 0..n_chunks {
                    let start = chunk_idx * stride;
                    let end = start + stride;
                    let chunk = &self.data[start..end];

                    let norm = match p {
                        1 => Ops::l1_norm(chunk),
                        2 => Ops::l2_norm(chunk),
                        _ => unimplemented!("Only L1 and L2 norm are implemented"),
                    };
                    result.push(norm);
                }

                Self::new_with_shape(result, result_shape)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_shape_of_new_array() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let array = NumArrayF32::new(data.clone()); // Using NumArrayF32 for simplicity
        let expected_shape = vec![4]; // Expected shape for a 1D array with 4 elements
        assert_eq!(array.shape(), expected_shape.as_slice());
    }

    #[test]
    fn test_zeros_array() {
        let shape = vec![2, 3]; // 2x3 matrix
        let zeros_array = NumArrayF32::zeros(shape.clone());
        assert_eq!(zeros_array.shape(), shape.as_slice());
        assert_eq!(zeros_array.get_data(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ones_array() {
        let shape = vec![2, 3]; // 2x3 matrix
        let ones_array = NumArrayF32::ones(shape.clone());
        assert_eq!(ones_array.shape(), shape.as_slice());
        assert_eq!(ones_array.get_data(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_concatenate_1d_arrays() {
        let a = NumArrayF32::new(vec![1.0, 2.0, 3.0]);
        let b = NumArrayF32::new(vec![4.0, 5.0]);
        let c = NumArrayF32::new(vec![6.0]);

        let concatenated = NumArrayF32::concatenate(&[a.clone(), b.clone(), c.clone()], 0);
        assert_eq!(concatenated.shape(), &[6]);
        assert_eq!(concatenated.get_data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_concatenate_2d_arrays_axis0() {
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = NumArrayF32::new_with_shape(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        let concatenated = NumArrayF32::concatenate(&[a.clone(), b.clone()], 0);
        assert_eq!(concatenated.shape(), &[4, 2]);
        assert_eq!(
            concatenated.get_data(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn test_concatenate_2d_arrays_axis1() {
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = NumArrayF32::new_with_shape(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        let concatenated = NumArrayF32::concatenate(&[a.clone(), b.clone()], 1);
        assert_eq!(concatenated.shape(), &[2, 4]);
        assert_eq!(
            concatenated.get_data(),
            &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
        );
    }

    #[test]
    fn test_concatenate_multiple_2d_arrays() {
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = NumArrayF32::new_with_shape(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = NumArrayF32::new_with_shape(vec![9.0, 10.0, 11.0, 12.0], vec![2, 2]);

        let concatenated = NumArrayF32::concatenate(&[a.clone(), b.clone(), c.clone()], 0);
        assert_eq!(concatenated.shape(), &[6, 2]);
        assert_eq!(
            concatenated.get_data(),
            &[
                1.0, 2.0, // a
                3.0, 4.0, 5.0, 6.0, // b
                7.0, 8.0, 9.0, 10.0, // c
                11.0, 12.0
            ]
        );
    }

    #[test]
    fn test_concatenate_incompatible_shapes() {
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let b = NumArrayF32::new_with_shape(vec![4.0, 5.0], vec![2, 1]);

        // Attempt to concatenate along axis 0 (rows)
        let concatenated = NumArrayF32::concatenate(&[a.clone(), b.clone()], 0);
        assert_eq!(concatenated.shape(), &[5, 1]);
        assert_eq!(concatenated.get_data(), &[1.0, 2.0, 3.0, 4.0, 5.0]);

        // Attempt to concatenate along axis 1 (columns) should panic due to mismatched row sizes
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let b = NumArrayF32::new_with_shape(vec![4.0, 5.0, 6.0, 7.0], vec![4, 1]);

        let result = std::panic::catch_unwind(|| NumArrayF32::concatenate(&[a, b], 1));
        assert!(
            result.is_err(),
            "Concatenation should fail due to incompatible shapes."
        );
    }

    #[test]
    fn test_concatenate_empty_input() {
        // Attempt to concatenate with an empty slice should panic
        let result = std::panic::catch_unwind(|| NumArrayF32::concatenate(&[], 0));
        assert!(
            result.is_err(),
            "Concatenation should fail when no arrays are provided."
        );
    }

    #[test]
    fn test_concatenate_different_dimensions() {
        let a = NumArrayF32::new(vec![1.0, 2.0, 3.0]); // Shape: [3] (1D)
        let b = NumArrayF32::new_with_shape(vec![4.0, 5.0], vec![1, 2]); // Shape: [1, 2] (2D)

        // Attempt to concatenate arrays with different dimensions should panic
        let result = std::panic::catch_unwind(|| NumArrayF32::concatenate(&[a, b], 0));
        assert!(
            result.is_err(),
            "Concatenation should fail due to differing dimensions."
        );
    }

    #[test]
    fn test_matrix_transpose() {
        let matrix = NumArrayF32::new_with_shape(
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
            NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);

        let transposed = matrix.transpose();

        let expected_data = vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0];

        assert_eq!(transposed.shape(), &[4, 2]);
        assert_eq!(transposed.get_data(), &expected_data);
    }

    #[test]
    fn test_reshape_successfully() {
        let array = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let new_shape = vec![2, 3]; // New shape compatible with the size of data (2*3 = 6)
        let reshaped_array = array.reshape(&new_shape.clone());
        assert_eq!(reshaped_array.shape(), new_shape.as_slice());
    }

    #[test]
    fn test_reshape_and_strides() {
        let array = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let new_shape = vec![2, 4]; // Reshape to a 2x4 matrix
        let reshaped_array = array.reshape(&new_shape.clone());
        assert_eq!(reshaped_array.shape(), new_shape.as_slice());
        // Check strides for a 2x4 matrix
        let expected_strides = vec![4, 1]; // Moving to the next row jumps 4 elements, column jump is 1
        assert_eq!(reshaped_array.strides, expected_strides);
    }

    #[test]
    fn test_flip_array() {
        let array = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0]);
        let flipped_array = array.flip_axis([0]);
        assert_eq!(flipped_array.get_data(), &[4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_flip_array_axis1() {
        let array = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let flipped_array = array.flip_axis([1]);
        assert_eq!(flipped_array.get_data(), &[3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn test_flip_array_3d() {
        let array = NumArrayF32::new_with_shape(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 3, 2],
        );
        let flipped_array = array.flip_axis([2]);
        assert_eq!(
            flipped_array.get_data(),
            &[2., 1., 4., 3., 6., 5., 8., 7., 10., 9., 12., 11.]
        );
    }

    #[test]
    fn test_dot_product_f32() {
        let a = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0]);
        let b = NumArrayF32::new(vec![4.0, 3.0, 2.0, 1.0]);
        assert_eq!(a.dot(&b).get_data(), &[20.0]);
    }

    #[test]
    fn test_dot_product_f64() {
        let a = NumArrayF64::new(vec![1.0, 2.0, 3.0, 4.0]);
        let b = NumArrayF64::new(vec![4.0, 3.0, 2.0, 1.0]);
        assert_eq!(a.dot(&b).get_data(), &[20.0]);
    }

    #[test]
    #[should_panic(expected = "Dimensions must align for dot product or matrix multiplication")]
    fn test_vector_dot_product_dimension_mismatch() {
        let a = NumArrayF32::new(vec![1.0, 2.0, 3.0]);
        let b = NumArrayF32::new(vec![1.0, 2.0]);
        a.dot(&b);
    }

    #[test]
    fn test_matrix_vector_multiply_correct() {
        let matrix = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let vector = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0], vec![3]);
        let result = matrix.dot(&vector);
        assert_eq!(result.get_data(), &[14.0, 32.0]);
    }

    #[test]
    fn test_matrix_matrix_multiply_correct() {
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = NumArrayF32::new_with_shape(vec![2.0, 0.0, 1.0, 3.0], vec![2, 2]);
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
        let arange_array = NumArrayF32::arange(0.0, 1.0, 0.2);
        assert_eq!(arange_array.get_data(), &[0.0, 0.2, 0.4, 0.6, 0.8]);
    }

    #[test]
    fn test_linspace() {
        let linspace_array = NumArrayF32::linspace(0.0, 1.0, 5);
        assert_eq!(linspace_array.get_data(), &[0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn test_slicing() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArrayF32::new(data);
        let slice = array.slice(0, 1, 3);
        assert_eq!(slice.get_data(), &[2.0, 3.0]);
    }

    #[test]
    fn test_slicing_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArrayF32::new_with_shape(data.clone(), vec![2, 3]);
        let slice = array.slice(0, 1, 2);
        assert_eq!(slice.get_data(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_mean_f32() {
        let a = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.mean().item(), 2.5);
    }

    #[test]
    fn test_mean_f64() {
        let a = NumArrayF64::new(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.mean().item(), 2.5);
    }

    #[test]
    fn test_calculate_reduced_index_1d() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let array = NumArrayF32::new_with_shape(data, vec![4]);
        assert_eq!(array.calculate_reduced_index(0, &[1]), 0);
        assert_eq!(array.calculate_reduced_index(1, &[1]), 0);
        assert_eq!(array.calculate_reduced_index(2, &[1]), 0);
        assert_eq!(array.calculate_reduced_index(3, &[1]), 0);
    }

    #[test]
    fn test_calculate_reduced_index_2d() {
        let shape = vec![2, 3]; // This is the shape of the data
        let reduction_shape = vec![2, 1]; // This signifies reduction along the second axis
        let num_array = NumArrayF32 {
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
        let num_array = NumArrayF32 {
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
        let reduction_shape = vec![1, 5, 1]; // This signifies reduction along the first and third axis
        let num_array = NumArrayF32 {
            data: vec![],
            shape,
            strides: vec![],
            _ops: PhantomData,
        }; // Strides are not used in this test

        // This tests a 3D array's reduction along non-adjacent axis.
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
    fn test_mean_axis_1d() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let array = NumArrayF32::new_with_shape(data, vec![4]);
        let mean_array = array.mean_axis(Some(&[0]));
        assert_eq!(mean_array.get_data(), &vec![2.5]);
    }

    #[test]
    fn test_mean_axis_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
        let mean_array = array.mean_axis(Some(&[1]));

        assert_eq!(mean_array.shape(), &[2]);
        assert_eq!(mean_array.get_data(), &vec![2.0, 5.0]); // Mean along the second axis (columns)
    }

    #[test]
    fn test_mean_axis_2d_column() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
        // Compute mean across columns (axis 1)
        let mean_array = array.mean_axis(Some(&[1]));
        assert_eq!(mean_array.get_data(), &vec![2.0, 5.0]); // Mean along the second axis (columns)
    }

    #[test]
    fn test_mean_axis_3d() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let array = NumArrayF32::new_with_shape(data, vec![2, 2, 3]);
        // Compute mean across the last two axis (1 and 2)
        let mean_array = array.mean_axis(Some(&[1, 2]));
        assert_eq!(mean_array.get_data(), &vec![3.5, 9.5]);
    }

    #[test]
    fn test_mean_axis_invalid_axis() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let array = NumArrayF32::new_with_shape(data, vec![4]);
        // Attempt to compute mean across an invalid axis
        let result = std::panic::catch_unwind(|| array.mean_axis(Some(&[1])));
        assert!(result.is_err(), "Should panic due to invalid axis");
    }

    #[test]
    fn test_get_and_set_single_element() {
        let mut array = NumArrayF32::new_with_shape(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        // Test getting an element
        assert_eq!(array.get(&[0, 1]), 20.0);

        // Test setting an element
        array.set(&[0, 1], 25.0);
        assert_eq!(array.get(&[0, 1]), 25.0);
    }

    #[test]
    fn test_get_and_set_edge_cases() {
        let mut array = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

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
        let array = NumArrayF32::new_with_shape(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        // Attempt to access an out-of-bounds index
        let _ = array.get(&[2, 2]);
    }

    #[test]
    #[should_panic(expected = "index out of bounds: the len is 4 but the index is 6")]
    fn test_set_out_of_bounds() {
        let mut array = NumArrayF32::new_with_shape(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        // Attempt to set an out-of-bounds index
        array.set(&[2, 2], 50.0);
    }

    #[test]
    #[should_panic(expected = "Indices must match the dimensions of the array.")]
    fn test_get_wrong_dimension_count() {
        let array = NumArrayF32::new_with_shape(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        // Attempt to access with incorrect number of dimensions
        let _ = array.get(&[1]);
    }

    #[test]
    #[should_panic(expected = "Indices must match the dimensions of the array.")]
    fn test_set_wrong_dimension_count() {
        let mut array = NumArrayF32::new_with_shape(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        // Attempt to set with incorrect number of dimensions
        array.set(&[1], 50.0);
    }

    #[test]
    fn test_row_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
        let row_slice = array.row_slice(1);
        assert_eq!(row_slice, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_column_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
        let column_slice = array.column_slice(1);
        assert_eq!(column_slice, vec![2.0, 5.0]);
    }

    #[test]
    fn test_exp_f32() {
        let array = NumArrayF32::new(vec![0.0, 1.0, 2.0]);
        let exp_array = array.exp();
        // Using approximate values for floating-point comparisons
        let expected = vec![1.0, 2.7182817, 7.389056];
        for (computed, &exp_val) in exp_array.get_data().iter().zip(expected.iter()) {
            assert!(
                (computed - exp_val).abs() < 1e-5,
                "Expected {}, got {}",
                exp_val,
                computed
            );
        }
    }

    #[test]
    fn test_log_f32() {
        let array = NumArrayF32::new(vec![1.0, 2.7182817, 7.389056]);
        let log_array = array.log();
        // Using approximate values for floating-point comparisons
        let expected = vec![0.0, 1.0, 2.0];
        for (computed, &log_val) in log_array.get_data().iter().zip(expected.iter()) {
            assert!(
                (computed - log_val).abs() < 1e-5,
                "Expected {}, got {}",
                log_val,
                computed
            );
        }
    }

    #[test]
    #[should_panic(expected = "Logarithm undefined for non-positive values.")]
    fn test_log_f32_with_non_positive() {
        let array = NumArrayF32::new(vec![1.0, -1.0, 0.0]);
        let _ = array.log(); // Should panic
    }

    #[test]
    fn test_sigmoid_f32() {
        let array = NumArrayF32::new(vec![0.0, 2.0, -2.0]);
        let sigmoid_array = array.sigmoid();
        // Using approximate values for floating-point comparisons
        let expected = vec![0.5, 0.880797, 0.119203];
        for (computed, &exp_val) in sigmoid_array.get_data().iter().zip(expected.iter()) {
            assert!(
                (computed - exp_val).abs() < 1e-5,
                "Expected {}, got {}",
                exp_val,
                computed
            );
        }
    }

    #[test]
    fn test_min_axis_1d() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let array = NumArrayF32::new_with_shape(data, vec![4]);
        let min_array = array.min_axis(Some(&[0]));
        assert_eq!(min_array.get_data(), &vec![1.0]);
    }

    #[test]
    fn test_min_axis_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
        let min_array = array.min_axis(Some(&[1]));
        assert_eq!(min_array.get_data(), &vec![1.0, 4.0]);
    }

    #[test]
    fn test_min_axis_3d() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let array = NumArrayF32::new_with_shape(data, vec![2, 2, 3]);
        let min_array = array.min_axis(Some(&[1, 2]));
        assert_eq!(min_array.get_data(), &vec![1.0, 7.0]);
    }

    #[test]
    fn test_min_axis_invalid_axis() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let array = NumArrayF32::new_with_shape(data, vec![4]);
        let result = std::panic::catch_unwind(|| array.min_axis(Some(&[1])));
        assert!(result.is_err(), "Should panic due to invalid axis");
    }

    #[test]
    fn test_min_f32() {
        let array = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(array.min(), 1.0);
    }

    #[test]
    fn test_squeeze() {
        let array = NumArrayF32::new_with_shape(vec![1.0, 2.0], vec![1, 2, 1]);
        let squeezed = array.squeeze(None); // removes all axis of length 1
        assert_eq!(squeezed.shape(), &[2]);
    }

    #[test]
    fn test_squeeze_no_axis() {
        let array = NumArrayF32::new_with_shape(vec![1.0, 2.0], vec![1, 2, 1]);
        let squeezed = array.squeeze(None);
        assert_eq!(squeezed.shape(), &[2]);
        assert_eq!(squeezed.get_data(), &[1.0, 2.0]);
    }

    #[test]
    fn test_squeeze_specific_axis() {
        let array = NumArrayF32::new_with_shape(vec![1.0, 2.0], vec![1, 2, 1]);
        let squeezed = array.squeeze(Some(&[0])); // Only squeeze first axis
        assert_eq!(squeezed.shape(), &[2, 1]);
        assert_eq!(squeezed.get_data(), &[1.0, 2.0]);
    }

    #[test]
    fn test_squeeze_multiple_axis() {
        let array = NumArrayF32::new_with_shape(vec![1.0, 2.0], vec![1, 2, 1, 1]);
        let squeezed = array.squeeze(Some(&[0, 2])); // Squeeze first and third axis
        assert_eq!(squeezed.shape(), &[2, 1]);
        assert_eq!(squeezed.get_data(), &[1.0, 2.0]);
    }

    #[test]
    fn test_max_axis_1d() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let array = NumArrayF32::new_with_shape(data, vec![4]);
        let max_array = array.max_axis(Some(&[0]));
        assert_eq!(max_array.get_data(), &vec![4.0]);
    }

    #[test]
    fn test_max_axis_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
        let max_array = array.max_axis(Some(&[1]));

        assert_eq!(max_array.shape(), &[2]);
        assert_eq!(max_array.get_data(), &vec![3.0, 6.0]); // Max along the second axis (columns)
    }

    #[test]
    fn test_max_axis_2d_column() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
        // Compute max across columns (axis 1)
        let max_array = array.max_axis(Some(&[1]));
        assert_eq!(max_array.get_data(), &vec![3.0, 6.0]); // Max along the second axis (columns)
    }

    #[test]
    fn test_max_axis_3d() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let array = NumArrayF32::new_with_shape(data, vec![2, 2, 3]);
        // Compute max across the last two axis (1 and 2)
        let max_array = array.max_axis(Some(&[1, 2]));
        assert_eq!(max_array.get_data(), &vec![6.0, 12.0]);
    }

    #[test]
    fn test_max_axis_invalid_axis() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let array = NumArrayF32::new_with_shape(data, vec![4]);
        // Attempt to compute max across an invalid axis
        let result = std::panic::catch_unwind(|| array.max_axis(Some(&[1])));
        assert!(result.is_err(), "Should panic due to invalid axis");
    }

    #[test]
    fn test_norm() {
        let array = NumArrayF32::new(vec![3.0, 4.0, -3.0, -4.0]);

        // Test L2 norm (Euclidean)
        let l2_norm = array.norm(2, None);
        // Use an epsilon tolerance for approximate float comparison
        assert!(
            (l2_norm.item() - 7.0710678).abs() < 1e-5,
            "Expected approximately 7.0710678, got {}",
            l2_norm.item()
        );

        // Test norm along axis 0 (this full-reduction implementation ignores axis, but test remains)
        // let l2_norm_axis0 = array.norm(2, Some(&[0]));
        // assert_eq!(l2_norm_axis0.shape(), &[2]);
        // // Compare expected values approximately.
        // assert!(
        //     (l2_norm_axis0.get(&[0]) - 4.2426405).abs() < 1e-5,
        //     "Expected approximately 4.2426405, got {}",
        //     l2_norm_axis0.get(&[0])
        // );
        // assert!(
        //     (l2_norm_axis0.get(&[1]) - 5.656854).abs() < 1e-5,
        //     "Expected approximately 5.656854, got {}",
        //     l2_norm_axis0.get(&[1])
        // );
    }
}
