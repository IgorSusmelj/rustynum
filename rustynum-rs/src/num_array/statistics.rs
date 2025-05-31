use crate::num_array::{NumArray, NumArrayF32, NumArrayF64};
use crate::simd_ops::SimdOps;
use crate::traits::{FromU32, FromUsize};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

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
    /// Sorts the array in ascending order.
    /// The original array is not modified.
    /// # Returns
    /// A new `NumArray` instance containing the sorted values.
    /// # Example
    /// ```
    /// use rustynum_rs::NumArrayF32;
    /// let data = vec![3.0, 1.0, 4.0, 2.0];
    /// let array = NumArrayF32::new(data);
    /// let sorted_array = array.sort();
    /// println!("Sorted array: {:?}", sorted_array.get_data());
    /// ```

    pub fn sort(&self) -> NumArray<T, Ops> {
        let mut sorted_data = self.data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        NumArray::new(sorted_data)
    }

    /// Computes the median of the array.
    /// The original array is not modified.
    ///
    /// # Returns
    /// A new `NumArray` instance containing the median value.
    ///
    /// # Examples
    /// ```
    /// use rustynum_rs::NumArrayF32;
    /// let data = vec![3.0, 1.0, 4.0, 2.0];
    /// let array = NumArrayF32::new(data);
    /// let median_array = array.median();
    /// println!("Median array: {:?}", median_array.get_data());
    /// ```
    ///

    pub fn median(&self) -> NumArray<T, Ops> {
        let sorted_data = self.sort();
        let median = Self::calculate_median(sorted_data.get_data());

        NumArray::new(vec![median])
    }
    /// Computes the median along the specified axis.
    /// The original array is not modified.
    ///
    /// # Parameters
    /// * `axis` - An optional reference to a vector of axis to compute the median along.
    ///
    /// # Returns
    /// A new `NumArray` instance containing the median values along the specified axis.
    ///
    /// # Panics
    /// Panics if any of the specified axis is out of bounds.
    ///
    /// # Examples
    /// ```
    /// use rustynum_rs::NumArrayF32;
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
    /// let median_array = array.median_axis(Some(&[1]));
    /// println!("Median array: {:?}", median_array.get_data());
    /// ```

    pub fn median_axis(&self, axis: Option<&[usize]>) -> NumArray<T, Ops> {
        match axis {
            Some(axis) => {
                let mut reduced_shape = self.shape.clone();
                let mut total_elements_to_reduce = 1;
                for &axis in axis {
                    assert!(axis < self.shape.len(), "Axis {} out of bounds.", axis);
                    reduced_shape[axis] = 1;
                    total_elements_to_reduce *= self.shape[axis];
                }

                let reduced_size: usize = reduced_shape.iter().product();
                let mut reduced_data = vec![T::from_u32(0); reduced_size];

                let mut accumulator = vec![T::from_u32(0); total_elements_to_reduce * reduced_size];

                let mut accumulator_ptrs: Vec<&mut [T]> =
                    accumulator.chunks_mut(total_elements_to_reduce).collect();
                let mut counts = vec![0; accumulator_ptrs.len()];

                for (i, &val) in self.data.iter().enumerate() {
                    let reduced_idx = self.calculate_reduced_index(i, &reduced_shape);
                    accumulator_ptrs[reduced_idx][counts[reduced_idx]] = val;
                    counts[reduced_idx] += 1;
                }

                for (i, ptr) in accumulator_ptrs.iter_mut().enumerate() {
                    ptr.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    reduced_data[i] = Self::calculate_median(ptr);
                }

                reduced_shape = reduced_shape
                    .into_iter()
                    .filter(|&x| x != 1)
                    .collect::<Vec<_>>();

                if reduced_shape.is_empty() {
                    return NumArray::new(reduced_data);
                } else {
                    return NumArray::new_with_shape(reduced_data, reduced_shape);
                }
            }
            None => self.median(),
        }
    }

    fn calculate_median(values: &[T]) -> T {
        let len = values.len();
        if len % 2 == 0 {
            (values[len / 2 - 1] + values[len / 2]) / T::from_u32(2)
        } else {
            values[len / 2]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_sort_f32() {
        let a = NumArrayF32::new(vec![5.0, 2.0, 3.0, 1.0, 4.0]);
        assert_eq!(a.sort().get_data(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_sort_f64() {
        let a = NumArrayF64::new(vec![5.0, 2.0, 3.0, 1.0, 4.0]);
        assert_eq!(a.sort().get_data(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_median_f32_one_elem() {
        let a = NumArrayF32::new(vec![1.0]);
        assert_eq!(a.median().item(), 1.0);
    }

    #[test]
    fn test_median_f32_even() {
        let a = NumArrayF32::new(vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0]);
        assert_eq!(a.median().item(), 3.5);
    }

    #[test]
    fn test_median_f32_uneven() {
        let a = NumArrayF32::new(vec![2.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(a.median().item(), 4.0);
    }

    #[test]
    fn test_median_f64_uneven() {
        let a = NumArrayF64::new(vec![1.0, 2.0, 2.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(a.median().item(), 4.0);
    }

    #[test]
    fn test_median_axis_1d_even() {
        let data = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0];
        let array = NumArrayF32::new_with_shape(data, vec![6]);
        let max_array = array.median_axis(Some(&[0]));
        assert_eq!(max_array.get_data(), &vec![3.5]);
    }

    #[test]
    fn test_median_axis_1d_uneven() {
        let data = vec![2.0, 2.0, 4.0, 3.0, 6.0, 5.0, 7.0];
        let array = NumArrayF32::new_with_shape(data, vec![7]);
        let max_array = array.median_axis(Some(&[0]));
        assert_eq!(max_array.get_data(), &vec![4.0]);
    }

    #[test]
    fn test_median_axis_2d_even() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let array = NumArrayF32::new_with_shape(data, vec![2, 3]);
        let max_array = array.median_axis(Some(&[0]));
        assert_eq!(max_array.get_data(), &vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_median_axis_2d_uneven() {
        let data = vec![2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let array = NumArrayF32::new_with_shape(data, vec![3, 3]);
        let max_array = array.median_axis(Some(&[1]));
        assert_eq!(max_array.get_data(), &vec![2.0, 5.0, 8.0]);
    }

    #[test]
    fn test_median_axis_3d_even() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let array = NumArrayF32::new_with_shape(data, vec![2, 2, 2]);
        let max_array = array.median_axis(Some(&[0, 1]));
        assert_eq!(max_array.get_data(), &vec![4.0, 5.0]);
    }

    #[test]
    fn test_median_axis_3d_uneven() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let array = NumArrayF32::new_with_shape(data, vec![3, 2, 2]);
        let max_array = array.median_axis(Some(&[0]));
        assert_eq!(max_array.get_data(), &vec![5.0, 6.0, 7.0, 8.0]);
    }
}
