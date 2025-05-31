// rustynum-rs/src/num_array/manipulation.rs
use super::NumArray;
use crate::simd_ops::SimdOps;
use crate::traits::{ExpLog, FromU32, FromUsize, NumOps};
use std::fmt::Debug;
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

impl<T, Ops> NumArray<T, Ops>
where
    T: Copy + Debug + Default + Clone,
{
    /// Transposes a 2D matrix from row-major to column-major format.
    /// The transpose operation reorders the data into a new contiguous `Vec<T>`.
    ///
    /// # Returns
    /// A new `NumArray` instance that is the transpose of the original matrix.
    /// with newly computed C-contiguous strides.
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
    Ops: SimdOps<T>, // SimdOps was part of the original block for these
{
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{num_array::NumArray, NumArrayF32};

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
    fn test_data_slice() {
        let data = vec![1.0, 2.0, 3.0];
        let array = NumArrayF32::new(data.clone());
        assert_eq!(array.data_slice(), data.as_slice());
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
}
