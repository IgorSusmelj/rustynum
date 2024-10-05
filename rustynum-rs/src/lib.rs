//! # RustyNum-Rs
//!
//! `rustynum-rs` is a numerical library for Rust, focusing on operations that can be vectorized using SIMD.
//! This crate provides efficient numerical arrays and operations, including basic arithmetic, dot products,
//! and transformations.

#![feature(array_chunks)]
#![feature(slice_as_chunks)]
#![feature(portable_simd)]

mod helpers;
pub mod num_array;
pub mod simd_ops;

pub mod traits;

pub use num_array::num_array::NumArrayF32;
pub use num_array::num_array::NumArrayF64;
pub use num_array::num_array::NumArrayI32;
pub use num_array::num_array::NumArrayI64;
pub use num_array::num_array::NumArrayU8;
