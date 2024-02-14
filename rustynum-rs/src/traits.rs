use std::ops::{Add, Div, Mul, Sub};

pub trait FromU32 {
    fn from_u32(value: u32) -> Self;
}

impl FromU32 for f32 {
    fn from_u32(value: u32) -> Self {
        value as f32
    }
}

impl FromU32 for f64 {
    fn from_u32(value: u32) -> Self {
        value as f64
    }
}

impl FromU32 for i32 {
    fn from_u32(value: u32) -> Self {
        value as i32
    }
}

impl FromU32 for i64 {
    fn from_u32(value: u32) -> Self {
        value as i64
    }
}

pub trait NumOps:
    Sized + Add<Output = Self> + Mul<Output = Self> + Sub<Output = Self> + Div<Output = Self> + Copy
{
    fn sqrt(self) -> Self;
    fn zero() -> Self;
}

impl NumOps for f32 {
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn zero() -> Self {
        0.0
    }
}

impl NumOps for f64 {
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn zero() -> Self {
        0.0
    }
}

impl NumOps for i32 {
    fn sqrt(self) -> Self {
        (self as f64).sqrt() as i32
    }
    fn zero() -> Self {
        0
    }
}

impl NumOps for i64 {
    fn sqrt(self) -> Self {
        (self as f64).sqrt() as i64
    }
    fn zero() -> Self {
        0
    }
}
