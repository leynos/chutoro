//! Integration tests for public dataset recipe error helpers.

use std::error::Error;

use chutoro_bench_datasets::RecipeError;

#[test]
fn recipe_error_other_wraps_arbitrary_source() {
    #[derive(Debug, thiserror::Error)]
    #[error("inner")]
    struct Inner;

    #[derive(Debug, thiserror::Error)]
    #[error("outer")]
    struct Outer {
        #[source]
        source: Inner,
    }

    let error = RecipeError::other(Outer { source: Inner });

    assert_eq!(error.to_string(), "outer");
    assert_eq!(
        error
            .source()
            .expect("opaque error should preserve source")
            .to_string(),
        "inner",
    );
}

/// Keep the 32-byte `RecipeError` budget visible to test-suite reviewers.
///
/// `clippy::result_large_err` is denied workspace-wide, and
/// `chutoro-bench-datasets/src/error.rs` enforces the same bound in a `const`
/// assertion. That guard is invisible when reading `tests/`, so this test
/// mirrors it and names the budget in its failure message. The bound is an
/// upper limit, not an exact size, because a smaller representation is always
/// acceptable.
#[test]
fn recipe_error_size_budget_is_respected() {
    const BUDGET_BYTES: usize = 32;

    assert!(
        std::mem::size_of::<RecipeError>() <= BUDGET_BYTES,
        "RecipeError must stay within its {BUDGET_BYTES}-byte budget: \
         `clippy::result_large_err` is denied workspace-wide and the \
         compile-time guard in `src/error.rs` enforces the same limit"
    );
}
