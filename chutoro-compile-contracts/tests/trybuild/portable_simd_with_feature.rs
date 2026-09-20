//! Compile-pass test: using std::simd with nightly_portable_simd feature enabled.
//!
//! This test verifies that when `nightly_portable_simd` feature IS enabled
//! (and running on nightly), code using std::simd::Simd compiles successfully.

#![cfg_attr(
    all(feature = "nightly_portable_simd", nightly),
    feature(portable_simd)
)]

#[cfg(all(feature = "nightly_portable_simd", nightly))]
use std::simd::Simd;

fn main() {
    #[cfg(all(feature = "nightly_portable_simd", nightly))]
    {
        // This should compile successfully when both nightly_portable_simd
        // feature and nightly cfg are present, as the portable_simd unstable
        // feature gate is enabled by the cfg_attr above.
        let _vec: Simd<f32, 16> = Simd::splat(1.0);
    }
}
