#![expect(clippy::expect_used, reason = "tests require contextual panics")]
use chutoro_core::{DataSource, DataSourceError};
use chutoro_providers_text::TextSource;
use rstest::rstest;

#[rstest]
fn distance_returns_abs_char_count_diff() {
    let ds = TextSource::new("demo", vec!["a".into(), "bbb".into()]);
    let dist = ds.distance(0, 1).expect("distance must succeed");
    assert_eq!(dist, 2.0);
}

#[rstest]
fn distance_bounds_check() {
    let ds = TextSource::new("demo", vec!["a".into()]);
    let err = ds.distance(0, 1).expect_err("distance must check bounds");
    assert!(matches!(err, DataSourceError::OutOfBounds { index: 1 }));
}
