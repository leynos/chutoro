//! Integration tests for public dataset recipe error helpers.

use std::error::Error;

use chutoro_bench_datasets::{CleanupFailure, IntegrityFailure, Phase, PortName, RecipeError};
use rstest::{fixture, rstest};

/// Innermost adapter error reached through the source chain.
#[derive(Debug, thiserror::Error)]
#[error("inner")]
struct Inner;

/// Opaque outer error that wraps [`Inner`].
#[derive(Debug, thiserror::Error)]
#[error("outer")]
struct Outer {
    /// Source preserved by the transparent wrapper.
    #[source]
    source: Inner,
}

/// Driver-created cleanup error retaining an opaque phase failure.
#[fixture]
fn compound_cleanup_error() -> RecipeError {
    RecipeError::cleanup_with_phase_error(
        Phase::Prepare,
        RecipeError::other(Outer { source: Inner }),
        RecipeError::port(PortName::Storage, "cache locked"),
    )
}

#[test]
fn recipe_error_other_wraps_arbitrary_source() {
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

#[rstest]
fn cleanup_with_phase_error_reports_both_reasons(compound_cleanup_error: RecipeError) {
    assert_eq!(
        compound_cleanup_error.to_string(),
        concat!(
            "cleanup failed in phase Prepare: port Storage failed: cache locked; ",
            "original phase error: outer",
        ),
    );

    let cleanup_failure = compound_cleanup_error
        .cleanup_failure()
        .expect("driver-created cleanup error should retain both failures");
    assert_eq!(
        cleanup_failure.to_string(),
        "port Storage failed: cache locked; original phase error: outer",
    );
}

#[rstest]
fn cleanup_failure_accessors_expose_both_errors(compound_cleanup_error: RecipeError) {
    let cleanup_failure = compound_cleanup_error
        .cleanup_failure()
        .expect("driver-created cleanup error should retain both failures");

    let RecipeError::Port(cleanup_error) = cleanup_failure.cleanup_error() else {
        panic!("cleanup error should be a port failure");
    };
    assert_eq!(cleanup_error.port, PortName::Storage);
    assert_eq!(cleanup_error.reason.as_ref(), "cache locked");

    let RecipeError::Other(phase_error) = cleanup_failure.phase_error() else {
        panic!("phase error should preserve the original recipe error");
    };
    assert!(
        phase_error.downcast_ref::<Outer>().is_some(),
        "phase error should expose the boxed opaque error",
    );
}

#[rstest]
fn cleanup_failure_source_chain_reaches_phase_source(compound_cleanup_error: RecipeError) {
    let cleanup_failure = compound_cleanup_error
        .source()
        .expect("cleanup error should expose its source")
        .downcast_ref::<CleanupFailure>()
        .expect("cleanup source should be the driver payload");

    let phase_error = cleanup_failure
        .source()
        .expect("cleanup payload should expose the phase error")
        .downcast_ref::<RecipeError>()
        .expect("cleanup payload source should be the phase recipe error");

    let RecipeError::Other(boxed_phase_error) = phase_error else {
        panic!("phase error should preserve the original recipe error");
    };
    assert_eq!(
        boxed_phase_error
            .downcast_ref::<Outer>()
            .expect("phase error should expose the boxed opaque error")
            .to_string(),
        "outer",
    );
    assert!(
        phase_error
            .source()
            .expect("transparent phase error should forward its source")
            .is::<Inner>(),
        "source chain should reach the innermost adapter error",
    );
}

#[rstest]
#[case::two_argument_cleanup(RecipeError::cleanup(
    Phase::Fetch,
    std::io::Error::other("remove cache"),
))]
#[case::non_cleanup_variant(RecipeError::validate("bad manifest"))]
fn cleanup_failure_is_none_without_the_driver_payload(#[case] error: RecipeError) {
    assert!(error.cleanup_failure().is_none());
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
