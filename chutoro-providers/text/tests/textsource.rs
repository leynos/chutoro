#![expect(clippy::expect_used, reason = "tests require contextual panics")]
use std::io::Cursor;

use chutoro_core::{DataSource, DataSourceError};
use chutoro_providers_text::{TextProvider, TextProviderError};
use rstest::rstest;

#[rstest]
#[case("kitten", "sitting", 3.0)]
#[case("gumbo", "gambol", 2.0)]
#[case("", "", 0.0)]
#[case("naïve", "naive", 1.0)]
fn distance_returns_levenshtein(#[case] left: &str, #[case] right: &str, #[case] expected: f32) {
    let provider = TextProvider::new("demo", vec![left.to_owned(), right.to_owned()])
        .expect("provider must build");
    let dist = provider.distance(0, 1).expect("distance must succeed");
    assert_eq!(dist, expected);
    let reverse = provider.distance(1, 0).expect("distance must succeed");
    assert_eq!(reverse, expected);
}

#[rstest]
fn distance_bounds_check() {
    let provider = TextProvider::new("demo", vec!["a".into()]).expect("provider must build");
    let err = provider
        .distance(0, 1)
        .expect_err("distance must check bounds");
    assert!(matches!(err, DataSourceError::OutOfBounds { index: 1 }));
}

#[rstest]
#[case("alpha\nbeta\n", &["alpha", "beta"])]
#[case("carriage\r\nreturn\r\n", &["carriage", "return"])]
#[case("lonely", &["lonely"])]
fn try_from_reader_trims_newlines(#[case] raw: &str, #[case] expected: &[&str]) {
    let cursor = Cursor::new(raw);
    let provider = TextProvider::try_from_reader("demo", cursor).expect("provider must build");
    let items: Vec<&str> = provider.lines().iter().map(String::as_str).collect();
    assert_eq!(items, expected);
}

#[rstest]
fn try_from_reader_empty_input() {
    let err =
        TextProvider::try_from_reader("demo", Cursor::new("")).expect_err("empty input must fail");
    assert!(matches!(err, TextProviderError::EmptyInput));
}

#[rstest]
fn new_rejects_empty_collection() {
    let err = TextProvider::new("demo", Vec::new()).expect_err("empty input must fail");
    assert!(matches!(err, TextProviderError::EmptyInput));
}
