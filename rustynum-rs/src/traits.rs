//! # Traits Module
//!
//! The `traits` module defines essential traits and their implementations for numerical operations
//! in the RustyNum-Rs library. These traits are designed to extend basic types with numerical
//! capabilities that are commonly used throughout the library.
//!
//! ## Features
//!
//! - `FromU32` and `FromUsize`: Conversion traits for creating numeric types from `u32` and `usize`.
//! - `NumOps`: A trait encapsulating common numerical operations like square root and zero initialization,
//!   supporting both floating-point and integer calculations.
//!
//! ## Implementations
//!
//! Each trait is implemented for fundamental numeric types (`f32`, `f64`, `i32`, `i64`) to ensure
//! they can be seamlessly used with the `NumArray` structures for operations requiring conversions
//! and basic arithmetic.
//!
//! ```rust
//! use rustynum_rs::traits::{FromU32, NumOps};
//!
//! let x: f32 = FromU32::from_u32(100);
//! let sqrt_x = x.sqrt();
//! println!("The square root of {} is {}", x, sqrt_x);
//! ```

use std::ops::{Add, Div, Mul, Sub};

/// Trait for converting `u32` to other numeric types.
pub trait FromU32 {
    /// Converts a `u32` to another type.
    ///
    /// # Parameters
    /// * `value` - The `u32` value to convert.
    ///
    /// # Returns
    /// The converted value.
    fn from_u32(value: u32) -> Self;
}

/// Converts an unsigned 32-bit integer to a 32-bit floating-point number.
impl FromU32 for f32 {
    /// Converts the given `value` from an unsigned 32-bit integer to a 32-bit floating-point number.
    ///
    /// # Arguments
    ///
    /// * `value` - The unsigned 32-bit integer value to convert.
    ///
    /// # Returns
    ///
    /// The converted 32-bit floating-point number.
    fn from_u32(value: u32) -> Self {
        value as f32
    }
}

/// Converts an unsigned 32-bit integer to a `f64` value.
impl FromU32 for f64 {
    /// Converts the given `u32` value to a `f64` value.
    ///
    /// # Arguments
    ///
    /// * `value` - The `u32` value to convert.
    ///
    /// # Returns
    ///
    /// The converted `f64` value.
    fn from_u32(value: u32) -> Self {
        value as f64
    }
}

/// Converts an unsigned 32-bit integer to an unsigned 8-bit integer.
impl FromU32 for u8 {
    /// Converts the given `value` of type `u32` to an unsigned 8-bit integer (`u8`).
    ///
    /// # Arguments
    ///
    /// * `value` - The unsigned 32-bit integer value to convert.
    ///
    /// # Returns
    ///
    /// The converted unsigned 8-bit integer value.
    fn from_u32(value: u32) -> Self {
        value as u8
    }
}

/// Converts an unsigned 32-bit integer to a signed 32-bit integer.
impl FromU32 for i32 {
    /// Converts the given `value` of type `u32` to a signed 32-bit integer (`i32`).
    ///
    /// # Arguments
    ///
    /// * `value` - The unsigned 32-bit integer value to convert.
    ///
    /// # Returns
    ///
    /// The converted signed 32-bit integer value.
    fn from_u32(value: u32) -> Self {
        value as i32
    }
}

/// Converts a `u32` value to an `i64`.
impl FromU32 for i64 {
    /// Converts the given `u32` value to an `i64`.
    ///
    /// # Arguments
    ///
    /// * `value` - The `u32` value to convert.
    ///
    /// # Returns
    ///
    /// The converted `i64` value.
    fn from_u32(value: u32) -> Self {
        value as i64
    }
}

/// A trait for converting from `usize` to a specific type.
pub trait FromUsize {
    /// Converts a `usize` value to the implementing type.
    fn from_usize(value: usize) -> Self;
}

/// Converts a `usize` value to `f32`.
impl FromUsize for f32 {
    /// Converts a `usize` value to `f32`.
    ///
    /// # Arguments
    ///
    /// * `value` - The `usize` value to convert.
    ///
    /// # Returns
    ///
    /// The converted `f32` value.
    fn from_usize(value: usize) -> Self {
        value as f32
    }
}

/// Converts a `usize` value to `f64`.
impl FromUsize for f64 {
    /// Converts the given `usize` value to `f64`.
    ///
    /// # Arguments
    ///
    /// * `value` - The `usize` value to convert.
    ///
    /// # Returns
    ///
    /// The converted `f64` value.
    fn from_usize(value: usize) -> Self {
        value as f64
    }
}

/// Converts a `usize` value to `u8` value.
///
/// # Panics
///
/// Panics if the `usize` value is too large to fit into a `u8`.
impl FromUsize for u8 {
    /// Converts a `usize` value into a `u8` value.
    ///
    /// # Arguments
    ///
    /// * `value` - The `usize` value to convert.
    ///
    /// # Returns
    ///
    /// The converted `u8` value.
    fn from_usize(value: usize) -> Self {
        value.try_into().expect("usize value is too large for u8")
    }
}

/// Converts a `usize` value into an `i32` value.
///
/// # Panics
///
/// Panics if the `usize` value is too large to fit into an `i32`.
impl FromUsize for i32 {
    // Ensure that the usize value fits into i32
    fn from_usize(value: usize) -> Self {
        value.try_into().expect("usize value is too large for i32")
    }
}

/// Converts a `usize` value to `i64`.
impl FromUsize for i64 {
    /// Converts a `usize` value to `i64`.
    ///
    /// # Arguments
    ///
    /// * `value` - The `usize` value to convert.
    ///
    /// # Returns
    ///
    /// The converted `i64` value.
    fn from_usize(value: usize) -> Self {
        value as i64
    }
}

pub trait NumOps:
    Sized + Add<Output = Self> + Mul<Output = Self> + Sub<Output = Self> + Div<Output = Self> + Copy
{
    /// Calculates the square root of the value.
    fn sqrt(self) -> Self;

    /// Returns the zero value of the type.
    fn zero() -> Self;
}

/// Trait implementation for performing numeric operations on `f32`.
impl NumOps for f32 {
    /// Returns the square root of the `f32` value.
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    /// Returns the zero value for `f32`.
    fn zero() -> Self {
        0.0
    }
}

/// Trait implementation for performing numeric operations on `f64`.
impl NumOps for f64 {
    /// Returns the square root of the `f64` value.
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    /// Returns the zero value for `f64`.
    fn zero() -> Self {
        0.0
    }
}

/// Trait implementation for performing numeric operations on `i32`.
impl NumOps for i32 {
    /// Calculates the square root of the `i32` value.  
    fn sqrt(self) -> Self {
        (self as f64).sqrt() as i32
    }

    /// Returns the zero value for `i32`.
    fn zero() -> Self {
        0
    }
}

/// Trait implementation for performing numeric operations on `i64`.
impl NumOps for i64 {
    /// Calculates the square root of the `i64` value.
    fn sqrt(self) -> Self {
        (self as f64).sqrt() as i64
    }

    /// Returns the zero value for `i64`.
    fn zero() -> Self {
        0
    }
}

/// A trait that provides exponential and logarithmic operations.
pub trait ExpLog: Sized {
    /// Returns the exponential of the number.
    fn exp(self) -> Self;

    /// Returns the natural logarithm of the number.
    ///
    /// # Panics
    ///
    /// Panics if the number is non-positive, as the logarithm is undefined for such values.
    fn log(self) -> Self;
}

impl ExpLog for f32 {
    fn exp(self) -> Self {
        f32::exp(self)
    }

    fn log(self) -> Self {
        f32::ln(self)
    }
}

impl ExpLog for f64 {
    fn exp(self) -> Self {
        f64::exp(self)
    }

    fn log(self) -> Self {
        f64::ln(self)
    }
}

/// A trait to provide a generic absolute value operation.
/// Only types that support an intrinsic absolute value are implemented.
pub trait AbsOps {
    fn abs(self) -> Self;
}

impl AbsOps for f32 {
    fn abs(self) -> Self {
        f32::abs(self)
    }
}

impl AbsOps for f64 {
    fn abs(self) -> Self {
        f64::abs(self)
    }
}

impl AbsOps for i32 {
    fn abs(self) -> Self {
        i32::abs(self)
    }
}

impl AbsOps for i64 {
    fn abs(self) -> Self {
        i64::abs(self)
    }
}
