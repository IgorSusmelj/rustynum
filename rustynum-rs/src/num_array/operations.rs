use super::NumArray;
use crate::simd_ops::SimdOps;
use crate::traits::{ExpLog, FromU32, FromUsize, NumOps};
use std::fmt::Debug;

use std::iter::Sum;
use std::ops::{Add, Div, Mul, Neg, Sub};

impl<T, Ops> Add<T> for NumArray<T, Ops>
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
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        let result_data = self.get_data().iter().map(|&x| x + rhs).collect::<Vec<_>>();
        Self::new(result_data)
    }
}

impl<'a, T, Ops> Add<T> for &'a NumArray<T, Ops>
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
    Ops: SimdOps<T>, // Ensure Ops is appropriate for T
{
    type Output = NumArray<T, Ops>;

    fn add(self, rhs: T) -> Self::Output {
        let result_data = self.get_data().iter().map(|&x| x + rhs).collect::<Vec<_>>();
        NumArray::new(result_data)
    }
}

impl<T, Ops> Add for NumArray<T, Ops>
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
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let result_data: Vec<T> = self
            .get_data()
            .iter()
            .zip(rhs.get_data().iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Self::new(result_data)
    }
}

impl<'a, 'b, T, Ops> Add<&'b NumArray<T, Ops>> for &'a NumArray<T, Ops>
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
    type Output = NumArray<T, Ops>;

    fn add(self, rhs: &'b NumArray<T, Ops>) -> Self::Output {
        let result_data = self
            .get_data()
            .iter()
            .zip(rhs.get_data().iter())
            .map(|(&x, &y)| x + y)
            .collect::<Vec<T>>();
        NumArray::new(result_data)
    }
}

impl<T, Ops> Sub<T> for NumArray<T, Ops>
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
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        let result_data = self.get_data().iter().map(|&x| x - rhs).collect::<Vec<_>>();
        Self::new(result_data)
    }
}

impl<'a, T, Ops> Sub<T> for &'a NumArray<T, Ops>
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
    Ops: SimdOps<T>, // Ensure Ops is appropriate for T
{
    type Output = NumArray<T, Ops>;

    fn sub(self, rhs: T) -> Self::Output {
        let result_data = self.get_data().iter().map(|&x| x - rhs).collect::<Vec<_>>();
        NumArray::new(result_data)
    }
}

impl<T, Ops> Sub for NumArray<T, Ops>
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
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let result_data: Vec<T> = self
            .get_data()
            .iter()
            .zip(rhs.get_data().iter())
            .map(|(&x, &y)| x - y)
            .collect();
        Self::new(result_data)
    }
}

impl<'a, 'b, T, Ops> Sub<&'b NumArray<T, Ops>> for &'a NumArray<T, Ops>
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
    type Output = NumArray<T, Ops>;

    fn sub(self, rhs: &'b NumArray<T, Ops>) -> Self::Output {
        let result_data = self
            .get_data()
            .iter()
            .zip(rhs.get_data().iter())
            .map(|(&x, &y)| x - y)
            .collect::<Vec<T>>();
        NumArray::new(result_data)
    }
}

impl<T, Ops> Mul<T> for NumArray<T, Ops>
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
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let result_data = self.get_data().iter().map(|&x| x * rhs).collect::<Vec<_>>();
        Self::new(result_data)
    }
}

impl<'a, T, Ops> Mul<T> for &'a NumArray<T, Ops>
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
    Ops: SimdOps<T>, // Ensure Ops is appropriate for T
{
    type Output = NumArray<T, Ops>;

    fn mul(self, rhs: T) -> Self::Output {
        let result_data = self.get_data().iter().map(|&x| x * rhs).collect::<Vec<_>>();
        NumArray::new(result_data)
    }
}

impl<T, Ops> Mul for NumArray<T, Ops>
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
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let result_data: Vec<T> = self
            .get_data()
            .iter()
            .zip(rhs.get_data().iter())
            .map(|(&x, &y)| x * y)
            .collect();
        Self::new(result_data)
    }
}

impl<'a, 'b, T, Ops> Mul<&'b NumArray<T, Ops>> for &'a NumArray<T, Ops>
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
    type Output = NumArray<T, Ops>;

    fn mul(self, rhs: &'b NumArray<T, Ops>) -> Self::Output {
        let result_data = self
            .get_data()
            .iter()
            .zip(rhs.get_data().iter())
            .map(|(&x, &y)| x * y)
            .collect::<Vec<T>>();
        NumArray::new(result_data)
    }
}

impl<T, Ops> Div<T> for NumArray<T, Ops>
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
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        let result_data = self.get_data().iter().map(|&x| x / rhs).collect::<Vec<_>>();
        Self::new(result_data)
    }
}

impl<'a, T, Ops> Div<T> for &'a NumArray<T, Ops>
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
    Ops: SimdOps<T>, // Ensure Ops is appropriate for T
{
    type Output = NumArray<T, Ops>;

    fn div(self, rhs: T) -> Self::Output {
        let result_data = self.get_data().iter().map(|&x| x / rhs).collect::<Vec<_>>();
        NumArray::new(result_data)
    }
}

impl<T, Ops> Div for NumArray<T, Ops>
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
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let result_data: Vec<T> = self
            .get_data()
            .iter()
            .zip(rhs.get_data().iter())
            .map(|(&x, &y)| x / y)
            .collect();
        Self::new(result_data)
    }
}

impl<'a, 'b, T, Ops> Div<&'b NumArray<T, Ops>> for &'a NumArray<T, Ops>
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
    type Output = NumArray<T, Ops>;

    fn div(self, rhs: &'b NumArray<T, Ops>) -> Self::Output {
        // Same shape case - use existing implementation
        if self.shape() == rhs.shape() {
            let result_data = self
                .get_data()
                .iter()
                .zip(rhs.get_data().iter())
                .map(|(&x, &y)| x / y)
                .collect::<Vec<T>>();
            return NumArray::new_with_shape(result_data, self.shape().to_vec());
        }

        // Broadcasting case for 2D arrays: self: [m, n], rhs: [m, 1]
        if self.shape().len() == 2
            && rhs.shape().len() == 2
            && self.shape()[0] == rhs.shape()[0]
            && rhs.shape()[1] == 1
        {
            let (m, n) = (self.shape()[0], self.shape()[1]);
            let mut result_data = Vec::with_capacity(m * n);

            for i in 0..m {
                let divisor = rhs.get(&[i, 0]);
                for j in 0..n {
                    result_data.push(self.get(&[i, j]) / divisor);
                }
            }
            // Important: maintain the original shape for the result
            return NumArray::new_with_shape(result_data, vec![m, n]);
        }

        panic!(
            "Shapes not broadcastable for division: {:?} vs {:?}",
            self.shape(),
            rhs.shape()
        );
    }
}

#[cfg(test)]
mod tests {

    use crate::{NumArrayF32, NumArrayF64, NumArrayI32, NumArrayI64, NumArrayU8};

    use super::*;

    #[test]
    fn test_add_scalar_u8() {
        let a = NumArrayU8::new(vec![1, 2, 3, 4]);
        let result = a + 1;
        assert_eq!(result.get_data(), &vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_add_array_u8() {
        let a = NumArrayU8::new(vec![1, 2, 3, 4]);
        let b = NumArrayU8::new(vec![4, 3, 2, 1]);
        let result = a + b;
        assert_eq!(result.get_data(), &vec![5, 5, 5, 5]);
    }

    #[test]
    fn test_add_scalar_i32() {
        let a = NumArrayI32::new(vec![1, 2, 3, 4]);
        let result = a + 1;
        assert_eq!(result.get_data(), &vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_add_scalar_i64() {
        let a = NumArrayI64::new(vec![1, 2, 3, 4]);
        let result = a + 1;
        assert_eq!(result.get_data(), &vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_add_scalar_f32() {
        let a = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0]);
        let result = a + 1.0;
        assert_eq!(result.get_data(), &vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_add_scalar_f64() {
        let a = NumArrayF64::new(vec![1.0, 2.0, 3.0, 4.0]);
        let result = a + 1.0;
        assert_eq!(result.get_data(), &vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_add_array_f32() {
        let a = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0]);
        let b = NumArrayF32::new(vec![4.0, 3.0, 2.0, 1.0]);
        let result = a + b;
        assert_eq!(result.get_data(), &vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_add_array_f64() {
        let a = NumArrayF64::new(vec![1.0, 2.0, 3.0, 4.0]);
        let b = NumArrayF64::new(vec![4.0, 3.0, 2.0, 1.0]);
        let result = a + b;
        assert_eq!(result.get_data(), &vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_add_scalar_with_remainder() {
        let data = (0..18).map(|x| x as f32).collect::<Vec<_>>(); // Length not divisible by 16 (assuming f32x16)
        let num_array = NumArrayF32::new(data);
        let scalar = 1.0f32;

        let result = num_array + scalar;

        // Check that the result has the correct length
        assert_eq!(result.get_data().len(), 18);

        // Check that each element in the result has been correctly incremented by the scalar
        for (i, &val) in result.get_data().iter().enumerate() {
            assert_eq!(val, i as f32 + scalar);
        }
    }

    #[test]
    fn test_add_arrays_with_remainder() {
        let data_a = (0..18).map(|x| x as f32).collect::<Vec<_>>(); // Length not divisible by 16
        let data_b = (0..18).map(|x| 2.0 * x as f32).collect::<Vec<_>>();

        let num_array_a = NumArrayF32::new(data_a);
        let num_array_b = NumArrayF32::new(data_b);

        let result = num_array_a + num_array_b;

        // Check that the result has the correct length
        assert_eq!(result.get_data().len(), 18);

        // Check that each element in the result is the sum of elements from the original arrays
        for (i, &val) in result.get_data().iter().enumerate() {
            assert_eq!(val, i as f32 + 2.0 * i as f32);
        }
    }

    #[test]
    fn test_broadcast_division() {
        // Test for f32
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = NumArrayF32::new_with_shape(vec![2.0, 4.0], vec![2, 1]);
        let result = &a / &b;
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.get_data(), &[0.5, 1.0, 1.5, 1.0, 1.25, 1.5]);

        // Test for f64
        let a = NumArrayF64::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = NumArrayF64::new_with_shape(vec![2.0, 4.0], vec![2, 1]);
        let result = &a / &b;
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.get_data(), &[0.5, 1.0, 1.5, 1.0, 1.25, 1.5]);
    }

    #[test]
    #[should_panic(expected = "Shapes not broadcastable")]
    fn test_invalid_broadcast_division() {
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = NumArrayF32::new_with_shape(vec![2.0], vec![1, 1]);
        let _result = &a / &b; // Should panic
    }

    #[test]
    fn test_broadcast_division_shape_preservation() {
        let a = NumArrayF32::new_with_shape(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        );
        let b = NumArrayF32::new_with_shape(
            vec![14.0_f32.sqrt(), 77.0_f32.sqrt(), 194.0_f32.sqrt()],
            vec![3, 1],
        );
        let result = &a / &b;

        // Check shape is preserved
        assert_eq!(result.shape(), &[3, 3]);

        // Check first row values
        let expected_first_row = vec![
            1.0 / 14.0_f32.sqrt(),
            2.0 / 14.0_f32.sqrt(),
            3.0 / 14.0_f32.sqrt(),
        ];

        for i in 0..3 {
            assert!(
                (result.get(&[0, i]) - expected_first_row[i]).abs() < 1e-5,
                "Mismatch at position [0, {}]",
                i
            );
        }
    }
}
