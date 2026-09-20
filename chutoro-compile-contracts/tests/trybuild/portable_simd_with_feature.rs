//! Compile-pass test: using std::simd with nightly_portable_simd feature enabled.
//!
//! This test verifies that nightly Rust enables `std::simd`.

#![feature(portable_simd)]

use std::simd::Simd;

fn main() {
    let _vec: Simd<f32, 16> = Simd::splat(1.0);
}
