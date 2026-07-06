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
