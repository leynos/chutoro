//! Compile-time tests for nightly portable-SIMD gating contracts.
//!
//! These trybuild tests verify that the stable/nightly feature matrix is enforced
//! at compile time for the portable-SIMD backend.

#[test]
#[cfg(not(nightly))]
fn portable_simd_is_rejected_without_feature() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/trybuild/portable_simd_without_feature.rs");
}

#[test]
#[cfg(all(nightly, feature = "nightly_portable_simd"))]
fn portable_simd_compiles_with_feature() {
    let t = trybuild::TestCases::new();
    t.pass("tests/trybuild/portable_simd_with_feature.rs");
}
