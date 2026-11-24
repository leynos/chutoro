//! Compile-pass contract for the compound cleanup-failure accessors.

use chutoro_bench_datasets::{CleanupFailure, Phase, PortName, RecipeError};

/// Both accessors are `const fn`, so they stay callable in a const context.
const fn accessors_are_const(failure: &CleanupFailure) -> (&RecipeError, &RecipeError) {
    (failure.phase_error(), failure.cleanup_error())
}

fn main() {
    let error = RecipeError::cleanup_with_phase_error(
        Phase::Prepare,
        RecipeError::prepare("archive corrupt"),
        RecipeError::port(PortName::Storage, "cache locked"),
    );
    let Some(failure) = error.cleanup_failure() else {
        return;
    };
    let (phase, cleanup) = accessors_are_const(failure);
    assert!(matches!(phase, RecipeError::Prepare(_)));
    assert_eq!(cleanup.to_string(), "port Storage failed: cache locked");
}