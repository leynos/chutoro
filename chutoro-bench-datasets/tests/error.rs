//! Integration tests for public dataset recipe error helpers.

use std::error::Error;

use chutoro_bench_datasets::{IntegrityFailure, RecipeError};

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

#[test]
fn integrity_failure_boxed_payload_preserves_source() {
    let failure = IntegrityFailure::InvalidHexCharacter { position: 63 };
    let expected = failure.to_string();
    let error = RecipeError::integrity(failure);

    assert_eq!(error.to_string(), format!("integrity failed: {expected}"));
    assert_eq!(
        error
            .source()
            .expect("integrity error should preserve source")
            .to_string(),
        expected,
    );
}

#[test]
fn recipe_error_size_budget_is_respected() {
    assert!(
        std::mem::size_of::<RecipeError>() <= 32,
        "RecipeError grew past its 32-byte budget: {}",
        std::mem::size_of::<RecipeError>(),
    );
}
