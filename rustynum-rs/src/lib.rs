#![feature(array_chunks)]
#![feature(slice_as_chunks)]
#![feature(portable_simd)]

mod num_array;
mod simd_ops;
mod traits;

pub use num_array::NumArray32;
pub use num_array::NumArray64;
