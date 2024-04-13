#[allow(unused_imports)]
use super::num_array::{NumArray, NumArray32, NumArray64};
use crate::simd_ops::SimdOps;
use crate::traits::{FromU32, FromUsize, NumOps};
use std::fmt::Debug;

use std::iter::Sum;
use std::ops::{Add, Div, Mul, Sub};

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
        + FromU32
        + FromUsize
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
        + FromU32
        + FromUsize
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
        + FromU32
        + FromUsize
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
        + FromU32
        + FromUsize
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
        + FromU32
        + FromUsize
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
        + FromU32
        + FromUsize
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
        + FromU32
        + FromUsize
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
        + FromU32
        + FromUsize
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
        + FromU32
        + FromUsize
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
        + FromU32
        + FromUsize
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
        + FromU32
        + FromUsize
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
        + FromU32
        + FromUsize
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
        + FromU32
        + FromUsize
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
        + FromU32
        + FromUsize
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
        + FromU32
        + FromUsize
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
        + FromU32
        + FromUsize
        + Debug,
    Ops: SimdOps<T>,
{
    type Output = NumArray<T, Ops>;

    fn div(self, rhs: &'b NumArray<T, Ops>) -> Self::Output {
        let result_data = self
            .get_data()
            .iter()
            .zip(rhs.get_data().iter())
            .map(|(&x, &y)| x / y)
            .collect::<Vec<T>>();
        NumArray::new(result_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// test for our newly added Add method
    #[test]
    fn test_add_scalar_f32() {
        let a = NumArray32::new(vec![1.0, 2.0, 3.0, 4.0]);
        let result = a + 1.0;
        assert_eq!(result.get_data(), &vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_add_scalar_f64() {
        let a = NumArray64::new(vec![1.0, 2.0, 3.0, 4.0]);
        let result = a + 1.0;
        assert_eq!(result.get_data(), &vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_add_array_f32() {
        let a = NumArray32::new(vec![1.0, 2.0, 3.0, 4.0]);
        let b = NumArray32::new(vec![4.0, 3.0, 2.0, 1.0]);
        let result = a + b;
        assert_eq!(result.get_data(), &vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_add_array_f64() {
        let a = NumArray64::new(vec![1.0, 2.0, 3.0, 4.0]);
        let b = NumArray64::new(vec![4.0, 3.0, 2.0, 1.0]);
        let result = a + b;
        assert_eq!(result.get_data(), &vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_add_scalar_with_remainder() {
        let data = (0..18).map(|x| x as f32).collect::<Vec<_>>(); // Length not divisible by 16 (assuming f32x16)
        let num_array = NumArray32::new(data);
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

        let num_array_a = NumArray32::new(data_a);
        let num_array_b = NumArray32::new(data_b);

        let result = num_array_a + num_array_b;

        // Check that the result has the correct length
        assert_eq!(result.get_data().len(), 18);

        // Check that each element in the result is the sum of elements from the original arrays
        for (i, &val) in result.get_data().iter().enumerate() {
            assert_eq!(val, i as f32 + 2.0 * i as f32);
        }
    }
}
