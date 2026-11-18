//! Compile-time contract checks for the stable error-code enums.
//!
//! `MstErrorCode`, `HierarchyErrorCode`, and `HnswErrorCode` are
//! `#[non_exhaustive]`, so downstream code must keep a wildcard arm when it
//! matches them. These fixtures pin the contract in both directions: an
//! exhaustive match fails to compile, and the wildcard form compiles.

/// Verifies the error-code enums require a wildcard arm downstream.
#[test]
fn error_codes_require_wildcard_arm() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/trybuild/error_code_wildcard_match.rs");
    cases.compile_fail("tests/trybuild/error_code_mst_exhaustive.rs");
    cases.compile_fail("tests/trybuild/error_code_hierarchy_exhaustive.rs");
    cases.compile_fail("tests/trybuild/error_code_hnsw_exhaustive.rs");
}
