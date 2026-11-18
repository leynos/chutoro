//! Integration tests for dataset recipe domain value types.

use chutoro_bench_datasets::{
    CacheKey, ObjectKey, RecipeError, RecipeVersion, SourceRole, SourceSpec, SourceUrl,
};
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

#[test]
fn opaque_keys_display_the_wrapped_identifier() {
    fn assert_display(key: impl AsRef<str> + std::fmt::Display, expected: &str) {
        assert_eq!(key.as_ref(), expected);
        assert_eq!(key.to_string(), expected);
        assert_eq!(format!("{key}"), expected);
    }

    assert_display(CacheKey::new("cache/a.bin"), "cache/a.bin");
    assert_display(ObjectKey::new("prepared/a.bin"), "prepared/a.bin");
}

#[test]
fn opaque_keys_accept_owned_strings() {
    let cache = CacheKey::new(String::from("cache/owned.bin"));
    let object = ObjectKey::new(String::from("prepared/owned.bin"));

    assert_eq!(cache.as_ref(), "cache/owned.bin");
    assert_eq!(object.to_string(), "prepared/owned.bin");
    assert_eq!(format!("{object}"), "prepared/owned.bin");
}
