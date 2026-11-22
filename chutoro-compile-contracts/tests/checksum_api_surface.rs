//! Compile-time contract checks for the checksum and integrity error surface.
//!
//! `Checksum` is an inhabited value and both `DigestAlgorithm` and
//! `IntegrityFailure` are non-exhaustive, so downstream code must keep a
//! wildcard arm when it matches either enum. The constructor and accessors
//! are usable in const contexts, while `Checksum::parse` is not, and the
//! removed `RecipeError::ChecksumUnsupported` variant must no longer compile.

/// Verifies the inhabited checksum value and boxed integrity errors compile.
///
/// The fixture also evaluates `Checksum::new`, `DigestAlgorithm::prefix`,
/// `DigestAlgorithm::digest_len`, `Checksum::algorithm`, and
/// `Checksum::as_bytes` in const contexts.
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

/// Verifies downstream matches keep a wildcard arm for `DigestAlgorithm`.
#[test]
fn digest_algorithm_requires_wildcard_arm() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/trybuild/digest_algorithm_exhaustive.rs");
}

/// Verifies parsing is rejected in a const context because it is not `const`.
#[test]
fn checksum_parse_is_not_const() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/trybuild/checksum_parse_in_const.rs");
}

/// Verifies the removed deferred-checksum variant no longer compiles.
#[test]
fn checksum_unsupported_variant_is_removed() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/trybuild/checksum_unsupported_variant_removed.rs");
}
