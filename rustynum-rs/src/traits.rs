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

pub trait NumOps: Sized + Add<Output = Self> + Mul<Output = Self> + Sub<Output = Self> + Div<Output = Self> + Copy {
  fn sqrt(self) -> Self;
  fn zero() -> Self;
}


impl NumOps for f32 {
  fn sqrt(self) -> Self { self.sqrt() }
  fn zero() -> Self { 0.0 }
}

impl NumOps for f64 {
  fn sqrt(self) -> Self { self.sqrt() }
  fn zero() -> Self { 0.0 }
}
