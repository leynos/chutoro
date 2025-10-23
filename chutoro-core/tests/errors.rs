//! Integration tests covering the public error types exposed by `chutoro-core`.

use std::{num::NonZeroUsize, sync::Arc};

use anyhow::{Context, Result};
use chutoro_core::{
    ChutoroError, ChutoroErrorCode, DataSourceError, DataSourceErrorCode, ExecutionStrategy,
};
use rstest::rstest;

type TestResult<T = ()> = Result<T>;

#[rstest]
#[case(DataSourceError::OutOfBounds { index: 0 }, DataSourceErrorCode::OutOfBounds)]
#[case(
    DataSourceError::OutputLengthMismatch { out: 1, expected: 2 },
    DataSourceErrorCode::OutputLengthMismatch,
)]
#[case(
    DataSourceError::DimensionMismatch { left: 1, right: 2 },
    DataSourceErrorCode::DimensionMismatch,
)]
#[case(DataSourceError::EmptyData, DataSourceErrorCode::EmptyData)]
#[case(DataSourceError::ZeroDimension, DataSourceErrorCode::ZeroDimension)]
fn returns_expected_data_source_code(
    #[case] error: DataSourceError,
    #[case] expected: DataSourceErrorCode,
) -> TestResult {
    assert_eq!(error.code(), expected);
    assert_eq!(error.code().as_str(), expected.as_str());
    Ok(())
}

#[test]
fn returns_expected_chutoro_code() -> TestResult {
    let min_cluster_size =
        NonZeroUsize::try_from(5).context("constant min_cluster_size must be non-zero")?;
    let cases: [(ChutoroError, ChutoroErrorCode, Option<DataSourceErrorCode>); 5] = [
        (
            ChutoroError::InvalidMinClusterSize { got: 0 },
            ChutoroErrorCode::InvalidMinClusterSize,
            None,
        ),
        (
            ChutoroError::EmptySource {
                data_source: Arc::from("empty"),
            },
            ChutoroErrorCode::EmptySource,
            None,
        ),
        (
            ChutoroError::InsufficientItems {
                data_source: Arc::from("small"),
                items: 3,
                min_cluster_size,
            },
            ChutoroErrorCode::InsufficientItems,
            None,
        ),
        (
            ChutoroError::BackendUnavailable {
                requested: ExecutionStrategy::Auto,
            },
            ChutoroErrorCode::BackendUnavailable,
            None,
        ),
        (
            ChutoroError::DataSource {
                data_source: Arc::from("source"),
                error: DataSourceError::OutOfBounds { index: 1 },
            },
            ChutoroErrorCode::DataSourceFailure,
            Some(DataSourceErrorCode::OutOfBounds),
        ),
    ];

    for (error, expected, data_source_code) in cases {
        assert_eq!(error.code(), expected);
        assert_eq!(error.code().as_str(), expected.as_str());
        assert_eq!(error.data_source_code(), data_source_code);
    }
    Ok(())
}
