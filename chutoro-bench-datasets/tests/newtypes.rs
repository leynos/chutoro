//! Integration tests for dataset recipe domain value types.

use chutoro_bench_datasets::{RecipeError, RecipeVersion, SourceRole, SourceSpec, SourceUrl};
use rstest::rstest;

mod common;
use common::expect_err;

#[rstest]
#[case::https("https://")]
#[case::s3("s3://")]
#[case::file("file://")]
#[case::ftp("ftp://example.test/x")]
fn source_url_rejects_invalid_scheme(#[case] value: &str) {
    let error = expect_err!(
        SourceUrl::parse(value),
        "invalid source URL should fail: {value}"
    );

    assert!(matches!(error, RecipeError::InvalidSource(_)));
}

#[rstest]
#[case::too_few_parts("1.2")]
#[case::too_many_parts("1.2.3.4")]
#[case::non_numeric("a.b.c")]
#[case::negative("-1.2.3")]
fn recipe_version_rejects_malformed_input(#[case] value: &str) {
    let error = expect_err!(
        RecipeVersion::parse(value),
        "malformed version should fail: {value}"
    );

    assert!(matches!(error, RecipeError::InvalidVersion(_)));
}

#[test]
fn source_spec_with_role_sets_role_and_no_checksum() {
    let url =
        SourceUrl::parse("https://example.test/file").expect("source URL fixture should parse");
    let spec = SourceSpec::with_role(url.clone(), SourceRole::Secondary);

    assert_eq!(spec.role, SourceRole::Secondary);
    assert_eq!(spec.url, url);
    assert!(spec.checksum.is_none());
}
