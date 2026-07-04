//! Integration tests for dataset recipe domain value types.

use chutoro_bench_datasets::{RecipeError, SourceUrl};
use rstest::rstest;

#[rstest]
#[case::https("https://")]
#[case::s3("s3://")]
#[case::file("file://")]
fn source_url_rejects_scheme_without_source_remainder(#[case] value: &str) {
    let Err(error) = SourceUrl::parse(value) else {
        panic!("scheme-only source URL should fail");
    };

    assert!(matches!(error, RecipeError::InvalidSource(_)));
}
