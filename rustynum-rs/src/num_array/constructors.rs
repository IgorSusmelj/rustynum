// rustynum-rs/src/num_array/constructors.rs
use super::NumArray; // To access the NumArray type from the parent module (defined in array_struct.rs via mod.rs)
use crate::simd_ops::SimdOps;
use crate::traits::{ExpLog, FromU32, FromUsize, NumOps}; // ExpLog might only be for linspace via Neg
use std::fmt::Debug;
use std::iter::Sum; // For Sum<T>
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub}; // Neg only for linspace via its original impl block

impl<T, Ops> NumArray<T, Ops>
where
    T: Clone
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Sum<T>
        +
        // Needed for original `zeros`/`ones` block, though not used by current `zeros`/`ones` impl
        NumOps
        + // Same as Sum<T>
        Copy
        + PartialOrd
        + FromU32
        + FromUsize
        + ExpLog
        + // ExpLog for `linspace`
        Neg<Output = T>
        + // For `linspace` (specifically, its original trait block had it)
        Default
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
        let num_elements = if shape.is_empty() {
            1 // Scalar case
        } else {
            shape.iter().product()
        };
        let data = vec![T::default(); num_elements];
        Self::new_with_shape(data, shape) // Relies on new_with_shape from array_struct.rs
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
        let num_elements = if shape.is_empty() {
            1 // Scalar case
        } else {
            shape.iter().product()
        };
        let data = vec![T::from_usize(1); num_elements]; // T must impl FromUsize
        Self::new_with_shape(data, shape)
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
    /// use rustynum_rs::NumArrayF32;
    /// let arange_array = NumArrayF32::arange(0.0, 1.0, 0.2);
    /// println!("Arange array: {:?}", arange_array.get_data());
    /// ```
    pub fn arange(start: T, stop: T, step: T) -> Self {
        if step == T::default() {
            panic!("step cannot be zero for arange.");
        }

        let mut data = Vec::new();
        let mut current = start;

        if step > T::default() {
            // Positive step
            // Ensure we don't overshoot due to floating point for positive step
            while current < stop {
                data.push(current);
                current = current + step;
            }
        } else {
            // Negative step
            // Ensure we don't overshoot due to floating point for negative step
            while current > stop {
                data.push(current);
                current = current + step;
            }
        }
        Self::new(data) // Relies on new() from array_struct.rs for 1D array
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
        if num == 0 {
            return Self::new(Vec::new());
        }
        if num == 1 {
            return Self::new(vec![start]);
        }

        let mut data = Vec::with_capacity(num);
        data.push(start); // First point is always start

        if num > 1 {
            // Only calculate step and intermediate points if more than 1 point
            // The divisor must be (num - 1) points.
            // T must support these operations accurately.
            let step = (stop - start) / T::from_usize(num - 1);
            for i in 1..(num - 1) {
                // Iterate for intermediate points
                data.push(start + T::from_usize(i) * step);
            }
            data.push(stop); // Last point is always stop
        }
        Self::new(data)
    }
}

#[cfg(test)]
mod tests {
    use crate::NumArrayF32;

    use super::*;

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
    fn test_arange() {
        let arange_array = NumArrayF32::arange(0.0, 1.0, 0.2);
        assert_eq!(arange_array.get_data(), &[0.0, 0.2, 0.4, 0.6, 0.8]);
    }

    #[test]
    fn test_linspace() {
        let linspace_array = NumArrayF32::linspace(0.0, 1.0, 5);
        assert_eq!(linspace_array.get_data(), &[0.0, 0.25, 0.5, 0.75, 1.0]);
    }
}
