//! Test support utilities for constructing Arrow arrays, Parquet data, and `DenseMatrixProvider`
//! instances from `RecordBatches`. Provides helpers for building `FixedSizeListArray` fixtures with
//! various null-handling patterns (row nulls, value nulls) and for serializing arrays to Parquet
//! format for ingest testing. Helpers are fallible so tests decide how failures are reported.

use super::{DenseMatrixProvider, DenseMatrixProviderError};
use crate::ingest::{append_fixed_size_list_values, validate_fixed_size_list_field};
use arrow_array::builder::BooleanBufferBuilder;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use bytes::Bytes;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::errors::ParquetError;
use std::{convert::TryFrom, iter, num::TryFromIntError, sync::Arc};

/// Failure modes of the Arrow and Parquet fixture builders.
///
/// Naming each source keeps the helpers self-documenting: a malformed fixture
/// (`RowLength`, `Dimension`) is distinguishable from an Arrow or Parquet
/// failure raised while assembling the fixture.
#[derive(Debug, thiserror::Error)]
pub(crate) enum FixtureError {
    /// A fixture row did not contain exactly `expected` values.
    #[error("row {row} has {actual} values but the fixture dimension is {expected}")]
    RowLength {
        /// Zero-based index of the offending row.
        row: usize,
        /// Fixture dimension every row must match.
        expected: usize,
        /// Number of values the row actually contained.
        actual: usize,
    },
    /// The dimension did not fit Arrow's `i32` fixed-size-list width.
    #[error("fixture dimension does not fit an i32 list width")]
    Dimension(#[from] TryFromIntError),
    /// Arrow rejected the record batch or schema.
    #[error(transparent)]
    Arrow(#[from] ArrowError),
    /// Parquet serialization failed.
    #[error(transparent)]
    Parquet(#[from] ParquetError),
}

/// Checks that a fixture row holds exactly `dimension` values.
///
/// # Errors
/// Returns [`FixtureError::RowLength`] when `actual` differs from `dimension`.
fn ensure_row_len(row: usize, actual: usize, dimension: usize) -> Result<(), FixtureError> {
    if actual == dimension {
        return Ok(());
    }
    Err(FixtureError::RowLength {
        row,
        expected: dimension,
        actual,
    })
}

/// Checks that every fixture row holds exactly `dimension` values.
///
/// # Errors
/// Returns [`FixtureError::RowLength`] for the first row whose length differs
/// from `dimension`, naming that row's index.
fn ensure_rows_len<T>(rows: &[Vec<T>], dimension: usize) -> Result<(), FixtureError> {
    for (index, row) in rows.iter().enumerate() {
        ensure_row_len(index, row.len(), dimension)?;
    }
    Ok(())
}

/// Builds a non-nullable three-dimensional fixed-size list array.
///
/// Convenience wrapper over [`build_list_array`] for the common three-feature
/// fixture shape; every row supplies exactly three values by construction.
///
/// # Errors
/// Propagates any [`FixtureError`] raised by [`build_list_array`].
pub(crate) fn build_array(rows: &[[f32; 3]]) -> Result<FixedSizeListArray, FixtureError> {
    let row_values = rows.iter().map(|row| row.to_vec()).collect::<Vec<_>>();
    build_list_array(&row_values, 3, false)
}

/// Builds a fixed-size list array from dense rows with no nulls.
///
/// Every row must contain exactly `dimension` values. `child_nullable` sets the
/// nullability flag on the list's item field, letting tests exercise schema
/// validation independently of the actual null buffer.
///
/// # Errors
/// Returns [`FixtureError::RowLength`] if any row length differs from
/// `dimension`, or [`FixtureError::Dimension`] if `dimension` exceeds `i32`.
pub(crate) fn build_list_array(
    rows: &[Vec<f32>],
    dimension: usize,
    child_nullable: bool,
) -> Result<FixedSizeListArray, FixtureError> {
    ensure_rows_len(rows, dimension)?;
    let values = Float32Array::from_iter_values(rows.iter().flatten().copied());
    fixed_size_list_from_values(values, dimension, child_nullable)
}

/// Builds a fixed-size list array where whole rows may be null.
///
/// `None` rows are recorded in the validity buffer and back-filled with zeroed
/// values so the child array keeps its fixed stride; `Some` rows must contain
/// exactly `dimension` values.
///
/// # Errors
/// Returns [`FixtureError::RowLength`] if a present row's length differs from
/// `dimension`, or [`FixtureError::Dimension`] if `dimension` exceeds `i32`.
pub(crate) fn build_list_array_with_row_nulls(
    rows: &[Option<Vec<f32>>],
    dimension: usize,
) -> Result<FixedSizeListArray, FixtureError> {
    let width = i32::try_from(dimension)?;
    let mut flat = Vec::with_capacity(rows.len().saturating_mul(dimension));
    let mut validity = BooleanBufferBuilder::new(rows.len());
    for (index, row) in rows.iter().enumerate() {
        if let Some(values) = row {
            ensure_row_len(index, values.len(), dimension)?;
            validity.append(true);
            flat.extend_from_slice(values);
        } else {
            validity.append(false);
            flat.extend(iter::repeat_n(0.0, dimension));
        }
    }
    let values = Float32Array::from(flat);
    Ok(FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        width,
        Arc::new(values) as ArrayRef,
        Some(validity.finish().into()),
    ))
}

/// Builds a fixed-size list array where individual values may be null.
///
/// Every row must contain exactly `dimension` entries, each of which may be
/// `None`; the resulting list marks its item field nullable.
///
/// # Errors
/// Returns [`FixtureError::RowLength`] if any row length differs from
/// `dimension`, or [`FixtureError::Dimension`] if `dimension` exceeds `i32`.
pub(crate) fn build_list_array_with_value_nulls(
    rows: &[Vec<Option<f32>>],
    dimension: usize,
) -> Result<FixedSizeListArray, FixtureError> {
    ensure_rows_len(rows, dimension)?;
    let values = rows.iter().flatten().copied().collect::<Float32Array>();
    fixed_size_list_from_values(values, dimension, true)
}

/// Builds the `features` schema field describing a fixed-size list column.
///
/// `child_nullable` and `list_nullable` control the item and list nullability
/// flags independently so tests can construct schemas that a provider should
/// accept or reject.
///
/// # Errors
/// Returns [`FixtureError::Dimension`] if `dimension` exceeds `i32`.
pub(crate) fn feature_field(
    dimension: usize,
    child_nullable: bool,
    list_nullable: bool,
) -> Result<Field, FixtureError> {
    Ok(Field::new(
        "features",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, child_nullable)),
            i32::try_from(dimension)?,
        ),
        list_nullable,
    ))
}

pub(crate) fn try_from_record_batches(
    name: impl Into<String>,
    column: &str,
    batches: Vec<RecordBatch>,
) -> Result<DenseMatrixProvider, DenseMatrixProviderError> {
    let mut values = Vec::new();
    let mut rows = 0_usize;
    let mut dimension: Option<usize> = None;

    for batch in batches {
        let schema = batch.schema();
        let index =
            schema
                .index_of(column)
                .map_err(|_| DenseMatrixProviderError::ColumnNotFound {
                    column: column.to_owned(),
                })?;
        let field = schema.field(index);
        let width = validate_fixed_size_list_field(field, column)?;
        if let Some(expected) = dimension {
            if expected != width {
                return Err(DenseMatrixProviderError::InconsistentBatchDimension {
                    expected,
                    actual: width,
                });
            }
        } else {
            dimension = Some(width);
        }
        let column_array = batch.column(index);
        let list = column_array
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| DenseMatrixProviderError::InvalidColumnType {
                column: column.to_owned(),
                actual: column_array.data_type().clone(),
            })?;
        append_fixed_size_list_values(list, dimension, rows, &mut values)?;
        rows += list.len();
    }

    let resolved_dimension = dimension.unwrap_or(0);
    Ok(DenseMatrixProvider::from_parts(
        name,
        rows,
        resolved_dimension,
        values,
    ))
}

/// Serializes a three-dimensional array to an in-memory Parquet file.
///
/// Uses the default non-nullable `features` field; call
/// [`write_parquet_with_field`] to control the schema.
///
/// # Errors
/// Returns [`FixtureError::Arrow`] if the array does not match the schema, or
/// [`FixtureError::Parquet`] if serialization fails.
pub(crate) fn write_parquet(array: FixedSizeListArray) -> Result<Bytes, FixtureError> {
    let field = feature_field(3, false, false)?;
    write_parquet_with_field(field, array)
}

/// Serializes a single array to an in-memory Parquet file under `field`.
///
/// `array` must match `field`'s declared type and nullability.
///
/// # Errors
/// Returns [`FixtureError::Arrow`] if the array does not match the schema, or
/// [`FixtureError::Parquet`] if serialization fails.
pub(crate) fn write_parquet_with_field(
    field: Field,
    array: FixedSizeListArray,
) -> Result<Bytes, FixtureError> {
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array) as ArrayRef])?;
    let mut buffer = Vec::new();
    {
        let mut writer = ArrowWriter::try_new(&mut buffer, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;
    }
    Ok(Bytes::from(buffer))
}

/// Serializes two arrays as separate row groups of one Parquet file.
///
/// Both arrays share `field`, exercising multi-batch ingest paths. Each must
/// match `field`'s declared type and nullability.
///
/// # Errors
/// Returns [`FixtureError::Arrow`] if either array does not match the schema,
/// or [`FixtureError::Parquet`] if serialization fails.
pub(crate) fn write_parquet_two_batches(
    first: FixedSizeListArray,
    second: FixedSizeListArray,
    field: Field,
) -> Result<Bytes, FixtureError> {
    let schema = Arc::new(Schema::new(vec![field]));
    let batch_one = RecordBatch::try_new(schema.clone(), vec![Arc::new(first) as ArrayRef])?;
    let batch_two = RecordBatch::try_new(schema.clone(), vec![Arc::new(second) as ArrayRef])?;
    let mut buffer = Vec::new();
    {
        let mut writer = ArrowWriter::try_new(&mut buffer, schema, None)?;
        writer.write(&batch_one)?;
        writer.write(&batch_two)?;
        writer.close()?;
    }
    Ok(Bytes::from(buffer))
}

/// Wraps a flat value array as a fixed-size list with the given stride.
///
/// # Errors
/// Returns [`FixtureError::Dimension`] if `dimension` exceeds `i32`.
fn fixed_size_list_from_values(
    values: Float32Array,
    dimension: usize,
    child_nullable: bool,
) -> Result<FixedSizeListArray, FixtureError> {
    Ok(FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, child_nullable)),
        i32::try_from(dimension)?,
        Arc::new(values) as ArrayRef,
        None,
    ))
}
