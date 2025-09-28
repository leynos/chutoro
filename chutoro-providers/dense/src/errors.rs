use arrow_schema::{ArrowError, DataType};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DenseMatrixProviderError {
    #[error("column `{column}` not found in Parquet schema")]
    ColumnNotFound { column: String },
    #[error("column `{column}` must be a FixedSizeList<Float32, _> but found {actual:?}")]
    InvalidColumnType { column: String, actual: DataType },
    #[error("FixedSizeList child type must be Float32 but found {actual:?}")]
    InvalidListValueType { actual: DataType },
    #[error("invalid FixedSizeList dimension {actual}")]
    InvalidDimension { actual: i32 },
    #[error("row {row} is null")]
    NullRow { row: usize },
    #[error("row {row} contains null value at position {value_index}")]
    NullValue { row: usize, value_index: usize },
    #[error("row {row} has length {actual} but expected {expected}")]
    InvalidRowLength {
        row: usize,
        expected: usize,
        actual: usize,
    },
    #[error("matrix with {rows} rows and dimension {dimension} exceeds capacity limits")]
    CapacityOverflow { rows: usize, dimension: usize },
    #[error("inconsistent dimensions across batches: expected {expected}, got {actual}")]
    InconsistentBatchDimension { expected: usize, actual: usize },
    #[error("arrow error: {0}")]
    Arrow(#[from] ArrowError),
    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
}
