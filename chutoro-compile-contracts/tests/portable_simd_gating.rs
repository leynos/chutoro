//! Compile-time tests for nightly portable-SIMD gating contracts.
//!
//! These trybuild tests verify that the stable/nightly feature matrix is enforced
//! at compile time for the portable-SIMD backend.

#[test]
fn portable_simd_gating_compile_checks() {
    let t = trybuild::TestCases::new();

    // When nightly_portable_simd feature is absent, portable-SIMD API should fail to compile
    t.compile_fail("tests/trybuild/portable_simd_without_feature.rs");

    // When nightly_portable_simd feature is present, portable-SIMD API should compile
    t.pass("tests/trybuild/portable_simd_with_feature.rs");
}
