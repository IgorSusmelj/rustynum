use std::ops::{Add, Mul, Div, Sub};

pub trait FromU32 {
  fn from_u32(value: u32) -> Self;
}

// Implement the trait for f32
impl FromU32 for f32 {
  fn from_u32(value: u32) -> Self {
      value as f32
  }
}

// Implement the trait for f64
impl FromU32 for f64 {
  fn from_u32(value: u32) -> Self {
      value as f64
  }
}

pub trait NumOps: Sized + Add<Output = Self> + Mul<Output = Self> + Sub<Output = Self> + Div<Output = Self> {
  fn sqrt(self) -> Self;
}

impl NumOps for f32 {
  fn sqrt(self) -> Self {
      f32::sqrt(self)
  }
}

impl NumOps for f64 {
  fn sqrt(self) -> Self {
      f64::sqrt(self)
  }
}