//! Errors emitted by dense matrix ingestion flows.
use arrow_schema::{ArrowError, DataType};
use thiserror::Error;

/// Variants cover the errors encountered when materializing dense matrices
/// from Arrow or Parquet sources so the public ingestion API remains documented.
#[derive(Debug, Error)]
pub enum DenseMatrixProviderError {
    /// A referenced column does not exist in the Arrow schema.
    #[error("column `{column}` not found in Parquet schema")]
    ColumnNotFound {
        /// Name of the column that was missing from the schema.
        column: String,
    },
    /// The Arrow column has an unexpected data type.
    #[error("column `{column}` must be a FixedSizeList<Float32, _> but found {actual:?}")]
    InvalidColumnType {
        /// Name of the offending column.
        column: String,
        /// Actual Arrow data type encountered at runtime.
        actual: DataType,
    },
    #[error(
        "column `{column}` or its child field must not be nullable (nullable_child: {nullable_child})"
    )]
    /// A column or its child elements were nullable but the provider requires
    /// non-null data to construct a dense matrix.
    NullableField {
        /// Name of the column that was nullable.
        column: String,
        /// Indicates whether the nested child field was nullable.
        nullable_child: bool,
    },
    /// Fixed-size list columns must contain 32-bit floating point child values.
    #[error("FixedSizeList child type must be Float32 but found {actual:?}")]
    InvalidListValueType {
        /// Actual child type discovered in the Arrow schema.
        actual: DataType,
    },
    /// Fixed-size list column declared an invalid dimension.
    #[error("invalid FixedSizeList dimension {actual}")]
    InvalidDimension {
        /// Dimension reported by the Arrow schema.
        actual: i32,
    },
    /// Encountered a completely null row within the dataset.
    #[error("row {row} is null")]
    NullRow {
        /// Index of the row containing only null values.
        row: usize,
    },
    /// Encountered a null element while reading a row.
    #[error("row {row} contains null value at position {value_index}")]
    NullValue {
        /// Index of the affected row.
        row: usize,
        /// Index within the row where the null value was observed.
        value_index: usize,
    },
    /// Row length did not match the expected dimensionality.
    #[error("row {row} has length {actual} but expected {expected}")]
    InvalidRowLength {
        /// Index of the row with mismatched length.
        row: usize,
        /// Required dimensionality for each row.
        expected: usize,
        /// Actual length encountered in the row.
        actual: usize,
    },
    /// Requested matrix size would exceed capacity constraints.
    #[error("matrix with {rows} rows and dimension {dimension} exceeds capacity limits")]
    CapacityOverflow {
        /// Number of rows attempted to be ingested.
        rows: usize,
        /// Dimensionality of each row.
        dimension: usize,
    },
    /// Batches within the same dataset reported conflicting dimensions.
    #[error("inconsistent dimensions across batches: expected {expected}, got {actual}")]
    InconsistentBatchDimension {
        /// Expected dimensionality inferred from earlier batches.
        expected: usize,
        /// Dimension reported by the current batch.
        actual: usize,
    },
    /// Wrapper around Arrow-specific ingestion failures.
    #[error("arrow error: {0}")]
    Arrow(#[from] ArrowError),
    /// Wrapper around Parquet-specific ingestion failures.
    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    /// Wrapper around standard I/O errors encountered during ingestion.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
}
