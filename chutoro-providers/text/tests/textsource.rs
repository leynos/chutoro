//! Integration tests covering the text-backed [`DataSource`] implementation.

use std::io::Cursor;

use anyhow::{Context, Result};
use chutoro_core::{DataSource, DataSourceError};
use chutoro_providers_text::{TextProvider, TextProviderError};
use rstest::rstest;

type TestResult<T = ()> = Result<T>;

#[rstest]
#[case("kitten", "sitting", 3.0)]
#[case("gumbo", "gambol", 2.0)]
#[case("", "", 0.0)]
#[case("naïve", "naive", 1.0)]
fn distance_returns_levenshtein(
    #[case] left: &str,
    #[case] right: &str,
    #[case] expected: f32,
) -> TestResult {
    let provider = TextProvider::new("demo", vec![left.to_owned(), right.to_owned()])
        .context("provider must build")?;
    let dist = provider.distance(0, 1).context("distance must succeed")?;
    assert_eq!(dist, expected);
    let reverse = provider.distance(1, 0).context("distance must succeed")?;
    assert_eq!(reverse, expected);

    for index in 0..provider.len() {
        let self_distance = provider
            .distance(index, index)
            .context("self-distance must succeed")?;
        assert_eq!(
            self_distance, 0.0,
            "distance({index}, {index}) must be zero"
        );
    }
    Ok(())
}

#[rstest]
fn distance_bounds_check() -> TestResult {
    let provider = TextProvider::new("demo", vec!["a".into()]).context("provider must build")?;
    let err = provider
        .distance(0, 1)
        .err()
        .context("distance must check bounds")?;
    assert!(matches!(err, DataSourceError::OutOfBounds { index: 1 }));
    Ok(())
}

#[rstest]
#[case("alpha\nbeta\n", &["alpha", "beta"])]
#[case("carriage\r\nreturn\r\n", &["carriage", "return"])]
#[case("lonely", &["lonely"])]
fn try_from_reader_trims_newlines(#[case] raw: &str, #[case] expected: &[&str]) -> TestResult {
    let cursor = Cursor::new(raw);
    let provider = TextProvider::try_from_reader("demo", cursor).context("provider must build")?;
    let items: Vec<&str> = provider.lines().iter().map(String::as_str).collect();
    assert_eq!(items, expected);
    Ok(())
}

#[rstest]
fn try_from_reader_empty_input() -> TestResult {
    let err = TextProvider::try_from_reader("demo", Cursor::new(""))
        .err()
        .context("empty input must fail")?;
    assert!(matches!(err, TextProviderError::EmptyInput));
    Ok(())
}

#[rstest]
fn try_from_reader_propagates_io_error() -> TestResult {
    struct FailingReader;

    impl std::io::Read for FailingReader {
        fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("boom"))
        }
    }

    impl std::io::BufRead for FailingReader {
        fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
            Err(std::io::Error::other("boom"))
        }

        fn consume(&mut self, _amt: usize) {}

        fn read_line(&mut self, _buf: &mut String) -> std::io::Result<usize> {
            Err(std::io::Error::other("boom"))
        }
    }

    let err = TextProvider::try_from_reader("demo", FailingReader)
        .err()
        .context("I/O failure must propagate")?;
    assert!(matches!(err, TextProviderError::Io(_)));
    Ok(())
}

#[rstest]
fn new_rejects_empty_collection() -> TestResult {
    let err = TextProvider::new("demo", Vec::new())
        .err()
        .context("empty input must fail")?;
    assert!(matches!(err, TextProviderError::EmptyInput));
    Ok(())
}

#[rstest]
fn data_source_reports_metadata() -> TestResult {
    let provider = TextProvider::new("demo", vec!["left".into(), "right".into()])
        .context("provider must build")?;
    assert_eq!(provider.name(), "demo");
    assert_eq!(provider.len(), 2);
    assert_eq!(provider.lines().len(), 2);
    let distance = provider
        .distance(0, 1)
        .context("distance calculation must succeed")?;
    assert_eq!(distance, 4.0);
    Ok(())
}
