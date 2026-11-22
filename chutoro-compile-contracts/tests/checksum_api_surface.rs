//! Compile-time contract checks for the checksum and integrity error surface.
//!
//! `Checksum` is an inhabited value and `IntegrityFailure` is non-exhaustive,
//! so downstream code must keep a wildcard arm when it matches the failure
//! type. The removed `RecipeError::ChecksumUnsupported` variant must no longer
//! compile.

/// Verifies the inhabited checksum value and boxed integrity errors compile.
#[test]
fn checksum_api_surface_compiles() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/trybuild/checksum_api_surface.rs");
}

/// Verifies downstream matches keep a wildcard arm for `IntegrityFailure`.
#[test]
fn integrity_failure_requires_wildcard_arm() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/trybuild/integrity_failure_exhaustive.rs");
}

/// Verifies the removed deferred-checksum variant no longer compiles.
#[test]
fn checksum_unsupported_variant_is_removed() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/trybuild/checksum_unsupported_variant_removed.rs");
}
