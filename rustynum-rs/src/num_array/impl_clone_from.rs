// rustynum-rs/src/num_array/impl_clone_from.rs
use super::NumArray;
use crate::simd_ops::SimdOps;
use crate::traits::NumOps;
use std::fmt::Debug;
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

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
