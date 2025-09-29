use super::DenseMatrixProviderError;
use arrow_schema::ArrowError;
use parquet::errors::ParquetError;
use rstest::rstest;
use std::io;

#[rstest]
#[case::arrow(
    DenseMatrixProviderError::from(ArrowError::ComputeError("boom".into())),
    true,
    false,
    false
)]
#[case::parquet(
    DenseMatrixProviderError::from(ParquetError::General("boom".into())),
    false,
    true,
    false
)]
#[case::io(
    DenseMatrixProviderError::from(io::Error::other("boom")),
    false,
    false,
    true
)]
fn dense_matrix_provider_error_conversions(
    #[case] err: DenseMatrixProviderError,
    #[case] is_arrow: bool,
    #[case] is_parquet: bool,
    #[case] is_io: bool,
) {
    assert_eq!(matches!(err, DenseMatrixProviderError::Arrow(_)), is_arrow);
    assert_eq!(
        matches!(err, DenseMatrixProviderError::Parquet(_)),
        is_parquet
    );
    assert_eq!(matches!(err, DenseMatrixProviderError::Io(_)), is_io);
}
