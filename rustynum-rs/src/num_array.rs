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
        let norm: T = self.data.iter().map(|&x| x * x).sum::<T>().sqrt();
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

impl<'a, T, Ops> From<&'a [T]> for NumArray<T, Ops>
where
    T: 'a + NumOps + Clone + Copy + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Sum<T>,
    Ops: SimdOps<T> + Default,
{
    fn from(data: &'a [T]) -> Self {
        Self { data: data.to_vec(), _ops: PhantomData }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

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
