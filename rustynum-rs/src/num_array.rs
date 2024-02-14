use std::ops::{Add, Mul, Div, Sub};
use std::iter::Sum;
use std::simd::{f32x16, f64x8};
use std::marker::PhantomData;

use crate::simd_ops::SimdOps;
use crate::traits::{FromU32, NumOps};

pub type NumArray32 = NumArray<f32, f32x16>;
pub type NumArray64 = NumArray<f64, f64x8>;

pub struct NumArray<T, Ops> {
    data: Vec<T>,
    _ops: PhantomData<Ops>,
}

impl<T, Ops> NumArray<T, Ops>
where
T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Sum<T> + NumOps + Copy + FromU32,
Ops: SimdOps<T>,
{
    pub fn new(data: Vec<T>) -> Self {
        Self {
            data,
            _ops: PhantomData,
        }
    }
    
    pub fn get_data(&self) -> &Vec<T> {
        &self.data
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
        Self::new(normalized_data)
    }
    

    pub fn slice(&self, start: usize, end: usize) -> Self {
        Self::new(self.data[start..end].to_vec())
    }
}

impl<T, Ops> From<Vec<T>> for NumArray<T, Ops> 
where
T: NumOps + Clone + Copy + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Sum<T>,
Ops: SimdOps<T> + Default, // Ensure Ops can be defaulted or appropriately initialized
{
    fn from(data: Vec<T>) -> Self {
        Self { data, _ops: PhantomData }
    }
}

impl<T, Ops> Clone for NumArray<T, Ops> 
where
    T: Clone,
{
    fn clone(&self) -> Self {
        NumArray {
            data: self.data.clone(),
            _ops: PhantomData,
        }
    }
}

impl<'a, T, Ops> From<&'a [T]> for NumArray<T, Ops>
where
T: 'a + NumOps + Clone + Copy + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Sum<T>,
Ops: SimdOps<T> + Default,
{
    fn from(data: &'a [T]) -> Self {
        Self { data: data.to_vec(), _ops: PhantomData }
    }
}

impl<T, Ops> Add<T> for NumArray<T, Ops>
where
    T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Sum<T> + NumOps + Copy + FromU32,
    Ops: SimdOps<T>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        let result_data = self.data.iter().map(|&x| x + rhs).collect::<Vec<_>>();
        Self::new(result_data)
    }
}

impl<'a, T, Ops> Add<T> for &'a NumArray<T, Ops>
where
    T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Sum<T> + NumOps + Copy + FromU32,
    Ops: SimdOps<T>, // Ensure Ops is appropriate for T
{
    type Output = NumArray<T, Ops>;

    fn add(self, rhs: T) -> Self::Output {
        let result_data = self.data.iter().map(|&x| x + rhs).collect::<Vec<_>>();
        NumArray::new(result_data)
    }
}


impl<T, Ops> Add for NumArray<T, Ops>
where
    T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Sum<T> + NumOps + Copy + FromU32,
    Ops: SimdOps<T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let result_data: Vec<T> = self.data.iter().zip(rhs.data.iter()).map(|(&x, &y)| x + y).collect();
        Self::new(result_data)
    }
}

impl<'a, 'b, T, Ops> Add<&'b NumArray<T, Ops>> for &'a NumArray<T, Ops>
where
    T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Sum<T> + NumOps + Copy + FromU32,
    Ops: SimdOps<T>,
{
    type Output = NumArray<T, Ops>;

    fn add(self, rhs: &'b NumArray<T, Ops>) -> Self::Output {
        let result_data = self.data.iter().zip(rhs.data.iter()).map(|(&x, &y)| x + y).collect::<Vec<T>>();
        NumArray::new(result_data)
    }
}


impl<T, Ops> Sub<T> for NumArray<T, Ops>
where
    T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Sum<T> + NumOps + Copy + FromU32,
    Ops: SimdOps<T>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        let result_data = self.data.iter().map(|&x| x - rhs).collect::<Vec<_>>();
        Self::new(result_data)
    }
}

impl<'a, T, Ops> Sub<T> for &'a NumArray<T, Ops>
where
    T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Sum<T> + NumOps + Copy + FromU32,
    Ops: SimdOps<T>, // Ensure Ops is appropriate for T
{
    type Output = NumArray<T, Ops>;

    fn sub(self, rhs: T) -> Self::Output {
        let result_data = self.data.iter().map(|&x| x - rhs).collect::<Vec<_>>();
        NumArray::new(result_data)
    }
}


impl<T, Ops> Sub for NumArray<T, Ops>
where
    T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Sum<T> + NumOps + Copy + FromU32,
    Ops: SimdOps<T>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let result_data: Vec<T> = self.data.iter().zip(rhs.data.iter()).map(|(&x, &y)| x - y).collect();
        Self::new(result_data)
    }
}

impl<'a, 'b, T, Ops> Sub<&'b NumArray<T, Ops>> for &'a NumArray<T, Ops>
where
    T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Sum<T> + NumOps + Copy + FromU32,
    Ops: SimdOps<T>,
{
    type Output = NumArray<T, Ops>;

    fn sub(self, rhs: &'b NumArray<T, Ops>) -> Self::Output {
        let result_data = self.data.iter().zip(rhs.data.iter()).map(|(&x, &y)| x - y).collect::<Vec<T>>();
        NumArray::new(result_data)
    }
}



impl<T, Ops> Mul<T> for NumArray<T, Ops>
where
    T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Sum<T> + NumOps + Copy + FromU32,
    Ops: SimdOps<T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let result_data = self.data.iter().map(|&x| x * rhs).collect::<Vec<_>>();
        Self::new(result_data)
    }
}

impl<'a, T, Ops> Mul<T> for &'a NumArray<T, Ops>
where
    T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Sum<T> + NumOps + Copy + FromU32,
    Ops: SimdOps<T>, // Ensure Ops is appropriate for T
{
    type Output = NumArray<T, Ops>;

    fn mul(self, rhs: T) -> Self::Output {
        let result_data = self.data.iter().map(|&x| x * rhs).collect::<Vec<_>>();
        NumArray::new(result_data)
    }
}


impl<T, Ops> Mul for NumArray<T, Ops>
where
    T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Sum<T> + NumOps + Copy + FromU32,
    Ops: SimdOps<T>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let result_data: Vec<T> = self.data.iter().zip(rhs.data.iter()).map(|(&x, &y)| x * y).collect();
        Self::new(result_data)
    }
}

impl<'a, 'b, T, Ops> Mul<&'b NumArray<T, Ops>> for &'a NumArray<T, Ops>
where
    T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Sum<T> + NumOps + Copy + FromU32,
    Ops: SimdOps<T>,
{
    type Output = NumArray<T, Ops>;

    fn mul(self, rhs: &'b NumArray<T, Ops>) -> Self::Output {
        let result_data = self.data.iter().zip(rhs.data.iter()).map(|(&x, &y)| x * y).collect::<Vec<T>>();
        NumArray::new(result_data)
    }
}


impl<T, Ops> Div<T> for NumArray<T, Ops>
where
    T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Sum<T> + NumOps + Copy + FromU32,
    Ops: SimdOps<T>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        let result_data = self.data.iter().map(|&x| x / rhs).collect::<Vec<_>>();
        Self::new(result_data)
    }
}

impl<'a, T, Ops> Div<T> for &'a NumArray<T, Ops>
where
    T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Sum<T> + NumOps + Copy + FromU32,
    Ops: SimdOps<T>, // Ensure Ops is appropriate for T
{
    type Output = NumArray<T, Ops>;

    fn div(self, rhs: T) -> Self::Output {
        let result_data = self.data.iter().map(|&x| x / rhs).collect::<Vec<_>>();
        NumArray::new(result_data)
    }
}


impl<T, Ops> Div for NumArray<T, Ops>
where
    T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Sum<T> + NumOps + Copy + FromU32,
    Ops: SimdOps<T>,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let result_data: Vec<T> = self.data.iter().zip(rhs.data.iter()).map(|(&x, &y)| x / y).collect();
        Self::new(result_data)
    }
}

impl<'a, 'b, T, Ops> Div<&'b NumArray<T, Ops>> for &'a NumArray<T, Ops>
where
    T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Sum<T> + NumOps + Copy + FromU32,
    Ops: SimdOps<T>,
{
    type Output = NumArray<T, Ops>;

    fn div(self, rhs: &'b NumArray<T, Ops>) -> Self::Output {
        let result_data = self.data.iter().zip(rhs.data.iter()).map(|(&x, &y)| x / y).collect::<Vec<T>>();
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
        let num_array = NumArray::<f32, f32x16>::new(data);
        let scalar = 1.0f32;

        let result = num_array + scalar;

        // Check that the result has the correct length
        assert_eq!(result.data.len(), 18);

        // Check that each element in the result has been correctly incremented by the scalar
        for (i, &val) in result.data.iter().enumerate() {
            assert_eq!(val, i as f32 + scalar);
        }
    }

    #[test]
    fn test_add_arrays_with_remainder() {
        let data_a = (0..18).map(|x| x as f32).collect::<Vec<_>>(); // Length not divisible by 16
        let data_b = (0..18).map(|x| 2.0 * x as f32).collect::<Vec<_>>();
        let num_array_a = NumArray::<f32, f32x16>::new(data_a);
        let num_array_b = NumArray::<f32, f32x16>::new(data_b);

        let result = num_array_a + num_array_b;

        // Check that the result has the correct length
        assert_eq!(result.data.len(), 18);

        // Check that each element in the result is the sum of elements from the original arrays
        for (i, &val) in result.data.iter().enumerate() {
            assert_eq!(val, i as f32 + 2.0 * i as f32);
        }
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
