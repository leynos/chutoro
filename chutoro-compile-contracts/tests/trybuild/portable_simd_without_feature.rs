//! Compile-fail test: attempting to use std::simd without nightly_portable_simd feature.
//!
//! This test verifies that when `nightly_portable_simd` feature is NOT enabled,
//! code attempting to use std::simd::Simd fails to compile because the
//! portable_simd unstable feature gate is not enabled.

use std::simd::Simd;

fn main() {
    // This should fail to compile because std::simd::Simd requires the
    // portable_simd unstable feature, which is only enabled when both
    // nightly_portable_simd feature and nightly cfg are present.
    let _vec: Simd<f32, 16> = Simd::splat(1.0);
}
