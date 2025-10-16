use std::{num::NonZeroUsize, sync::Arc};

use chutoro_core::{
    ChutoroError, ChutoroErrorCode, DataSourceError, DataSourceErrorCode, ExecutionStrategy,
};
use rstest::rstest;

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
) {
    assert_eq!(error.code(), expected);
    assert_eq!(error.code().as_str(), expected.as_str());
}

#[rstest]
#[case(
    ChutoroError::InvalidMinClusterSize { got: 0 },
    ChutoroErrorCode::InvalidMinClusterSize,
    None,
)]
#[case(
    ChutoroError::EmptySource { data_source: Arc::from("empty") },
    ChutoroErrorCode::EmptySource,
    None,
)]
#[case(
    ChutoroError::InsufficientItems {
        data_source: Arc::from("small"),
        items: 3,
        min_cluster_size: NonZeroUsize::new(5).expect("non-zero"),
    },
    ChutoroErrorCode::InsufficientItems,
    None,
)]
#[case(
    ChutoroError::BackendUnavailable {
        requested: ExecutionStrategy::Auto,
    },
    ChutoroErrorCode::BackendUnavailable,
    None,
)]
#[case(
    ChutoroError::DataSource {
        data_source: Arc::from("source"),
        error: DataSourceError::OutOfBounds { index: 1 },
    },
    ChutoroErrorCode::DataSourceFailure,
    Some(DataSourceErrorCode::OutOfBounds),
)]
fn returns_expected_chutoro_code(
    #[case] error: ChutoroError,
    #[case] expected: ChutoroErrorCode,
    #[case] data_source_code: Option<DataSourceErrorCode>,
) {
    assert_eq!(error.code(), expected);
    assert_eq!(error.code().as_str(), expected.as_str());
    assert_eq!(error.data_source_code(), data_source_code);
}
