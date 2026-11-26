//! Compile-time contracts for the compound cleanup-failure API.
//!
//! The pass fixture records that `CleanupFailure` is reachable only through
//! `RecipeError::cleanup_failure()` and that both accessors are `const fn`.
//! The compile-fail fixture records that its private fields cannot be built
//! from outside `chutoro-bench-datasets`.

/// Verifies the cleanup payload's construction and access contracts.
#[test]
fn cleanup_failure_api_surface_is_constrained() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/trybuild/cleanup_failure_api.rs");
    cases.compile_fail("tests/trybuild/cleanup_failure_private.rs");
}
