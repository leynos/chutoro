//! Compile-time tests for nightly portable-SIMD gating contracts.
//!
//! These trybuild tests verify that the stable/nightly feature matrix is enforced
//! at compile time for the portable-SIMD backend.

/// Confirms stable builds reject portable SIMD without its opt-in feature.
#[test]
#[cfg(not(nightly))]
fn portable_simd_is_rejected_without_feature() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/trybuild/portable_simd_without_feature.rs");
}

/// Confirms nightly builds accept portable SIMD when its feature is enabled.
#[test]
#[cfg(all(nightly, feature = "nightly_portable_simd"))]
fn portable_simd_compiles_with_feature() {
    let t = trybuild::TestCases::new();
    t.pass("tests/trybuild/portable_simd_with_feature.rs");
}
