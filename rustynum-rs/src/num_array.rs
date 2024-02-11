use crate::simd_ops::{dot_product_simd_f32, dot_product_simd_f64};

/// A simple numeric array for `f32` demonstration purposes.
pub struct NumArray32 {
    data: Vec<f32>,
}

/// A simple numeric array for `f64` demonstration purposes.
pub struct NumArray64 {
    data: Vec<f64>,
}

impl NumArray32 {
    /// Creates a new `NumArray32` from a slice of `f32`.
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn get_data(&self) -> &Vec<f32> {
        &self.data
    }

    // Normalizes the NumArray to unit norm.
    pub fn normalize(&self) -> NumArray32 {
        let norm = self.data.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let normalized_data = self.data.iter().map(|&x| x / norm).collect();
        NumArray32::new(normalized_data)
    }

    /// Calculates the dot product between two `NumArray32` instances.
    pub fn dot(&self, other: &Self) -> f32 {
        dot_product_simd_f32(&self.data, &other.data)
    }

    // Computes cosine similarity between two `NumArray` instances.
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let norm_self = self.normalize();
        let norm_other = other.normalize();
        norm_self.dot(&norm_other)
    }
}

impl NumArray64 {
    /// Creates a new `NumArray64` from a slice of `f64`.
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }

    pub fn get_data(&self) -> &Vec<f64> {
        &self.data
    }

    // Normalizes the NumArray to unit norm.
    pub fn normalize(&self) -> NumArray64 {
        let norm = self.data.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let normalized_data = self.data.iter().map(|&x| x / norm).collect();
        NumArray64::new(normalized_data)
    }

    /// Calculates the dot product between two `NumArray64` instances.
    pub fn dot(&self, other: &Self) -> f64 {
        dot_product_simd_f64(&self.data, &other.data)
    }

    // Computes cosine similarity between two `NumArray` instances.
    pub fn cosine_similarity(&self, other: &Self) -> f64 {
        let norm_self = self.normalize();
        let norm_other = other.normalize();
        norm_self.dot(&norm_other)
    }
}

// Implement the From trait for both `NumArray32` and `NumArray64`
// Similar to what is already defined for `NumArray32`, but for `NumArray64`
impl From<&[f32]> for NumArray32 {
    fn from(slice: &[f32]) -> Self {
        Self::new(slice.to_vec())
    }
}
impl From<Vec<f32>> for NumArray32 {
    fn from(vec: Vec<f32>) -> Self {
        Self { data: vec }
    }
}

impl From<&[f64]> for NumArray64 {
    fn from(slice: &[f64]) -> Self {
        Self::new(slice.to_vec())
    }
}
impl From<Vec<f64>> for NumArray64 {
    fn from(vec: Vec<f64>) -> Self {
        Self { data: vec }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_f32() {
        let a = NumArray32::new((&[1.0, 2.0, 3.0, 4.0]).to_vec());
        let b = NumArray32::new((&[4.0, 3.0, 2.0, 1.0]).to_vec());
        assert_eq!(a.dot(&b), 20.0);
    }

    #[test]
    fn test_dot_product_f64() {
        let a = NumArray64::new((&[1.0, 2.0, 3.0, 4.0]).to_vec());
        let b = NumArray64::new((&[4.0, 3.0, 2.0, 1.0]).to_vec());
        assert_eq!(a.dot(&b), 20.0);
    }
}
