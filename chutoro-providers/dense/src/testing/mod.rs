//! Shared Arrow and Parquet fixtures for dense-provider and dev-dependent tests.
//!
//! Enable the crate's `testing` feature to reuse these builders from another
//! crate's development dependencies. The module stays out of ordinary builds.

use std::{convert::TryFrom, io::Write, iter, num::TryFromIntError, sync::Arc};

use arrow_array::builder::BooleanBufferBuilder;
use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use bytes::Bytes;
use parquet::{arrow::arrow_writer::ArrowWriter, errors::ParquetError};

/// Failure modes of the Arrow and Parquet fixture builders.
#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
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

/// Builds a non-nullable fixed-size list array from rows with a shared width.
///
/// The const generic keeps literal fixture rows concise while supporting every
/// Arrow-compatible dimension.
///
/// # Examples
/// ```ignore
/// let array = build_array(&[[1.0, 2.0], [3.0, 4.0]])?;
/// assert_eq!(array.value_length(), 2);
/// assert_eq!(array.len(), 2);
/// # Ok::<(), FixtureError>(())
/// ```
///
/// # Errors
/// Returns [`FixtureError::Dimension`] if the width does not fit Arrow's
/// fixed-size-list representation.
pub fn build_array<const N: usize>(rows: &[[f32; N]]) -> Result<FixedSizeListArray, FixtureError> {
    let row_values = rows.iter().map(|row| row.to_vec()).collect::<Vec<_>>();
    build_list_array(&row_values, N, false)
}

/// Builds a fixed-size list array from dense rows with no nulls.
///
/// `child_nullable` sets the item field's schema nullability, allowing schema
/// validation tests to vary it independently of the array's validity buffer.
///
/// # Examples
/// ```ignore
/// let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
/// let array = build_list_array(&rows, 2, false)?;
/// assert_eq!(array.value_length(), 2);
/// # Ok::<(), FixtureError>(())
/// ```
///
/// # Errors
/// Returns [`FixtureError::RowLength`] if a row has the wrong width, or
/// [`FixtureError::Dimension`] if `dimension` exceeds `i32`.
pub fn build_list_array(
    rows: &[Vec<f32>],
    dimension: usize,
    child_nullable: bool,
) -> Result<FixedSizeListArray, FixtureError> {
    ensure_rows_len(rows, dimension)?;
    let values = Float32Array::from_iter_values(rows.iter().flatten().copied());
    fixed_size_list_from_values(values, dimension, child_nullable)
}

/// Builds a fixed-size list array where entire rows may be null.
///
/// `None` rows are marked in the outer validity buffer and padded with zeroes
/// so the child array retains its fixed stride.
///
/// # Examples
/// ```ignore
/// let rows = vec![Some(vec![1.0, 2.0]), None];
/// let array = build_list_array_with_row_nulls(&rows, 2)?;
/// assert!(array.is_null(1));
/// # Ok::<(), FixtureError>(())
/// ```
///
/// # Errors
/// Returns [`FixtureError::RowLength`] if a present row has the wrong width,
/// or [`FixtureError::Dimension`] if `dimension` exceeds `i32`.
pub fn build_list_array_with_row_nulls(
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

/// Builds a fixed-size list array where individual scalar values may be null.
///
/// Rows themselves are never null, and the item field is nullable.
///
/// # Examples
/// ```ignore
/// let rows = vec![vec![Some(1.0), None]];
/// let array = build_list_array_with_value_nulls(&rows, 2)?;
/// assert!(array.values().is_null(1));
/// # Ok::<(), FixtureError>(())
/// ```
///
/// # Errors
/// Returns [`FixtureError::RowLength`] if a row has the wrong width, or
/// [`FixtureError::Dimension`] if `dimension` exceeds `i32`.
pub fn build_list_array_with_value_nulls(
    rows: &[Vec<Option<f32>>],
    dimension: usize,
) -> Result<FixedSizeListArray, FixtureError> {
    ensure_rows_len(rows, dimension)?;
    let values = rows.iter().flatten().copied().collect::<Float32Array>();
    fixed_size_list_from_values(values, dimension, true)
}

/// Builds the `features` field for a fixed-size list column.
///
/// The item and outer-list nullability flags vary independently.
///
/// # Examples
/// ```ignore
/// let field = feature_field(2, true, false)?;
/// assert!(field.is_nullable() == false);
/// # Ok::<(), FixtureError>(())
/// ```
///
/// # Errors
/// Returns [`FixtureError::Dimension`] if `dimension` exceeds `i32`.
pub fn feature_field(
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

/// Serializes an array with three values per row to an in-memory Parquet file.
///
/// Use [`write_parquet_with_field`] to choose the field's dimension and
/// nullability explicitly.
///
/// # Examples
/// ```ignore
/// let array = build_array(&[[1.0, 2.0, 3.0]])?;
/// let parquet = write_parquet(array)?;
/// assert!(!parquet.is_empty());
/// # Ok::<(), FixtureError>(())
/// ```
///
/// # Errors
/// Returns [`FixtureError::Arrow`] if the array does not match the schema, or
/// [`FixtureError::Parquet`] if serialization fails.
pub fn write_parquet(array: FixedSizeListArray) -> Result<Bytes, FixtureError> {
    let field = feature_field(3, false, false)?;
    write_parquet_with_field(field, array)
}

/// Serializes an array to an in-memory Parquet file under `field`.
///
/// # Examples
/// ```ignore
/// let field = feature_field(2, false, false)?;
/// let array = build_array(&[[1.0, 2.0]])?;
/// let parquet = write_parquet_with_field(field, array)?;
/// assert!(!parquet.is_empty());
/// # Ok::<(), FixtureError>(())
/// ```
///
/// # Errors
/// Returns [`FixtureError::Arrow`] if the array does not match the schema, or
/// [`FixtureError::Parquet`] if serialization fails.
pub fn write_parquet_with_field(
    field: Field,
    array: FixedSizeListArray,
) -> Result<Bytes, FixtureError> {
    let (schema, batch) = record_batch(field, array)?;
    let mut buffer = Vec::new();
    write_record_batches(&mut buffer, schema, &[batch])?;
    Ok(Bytes::from(buffer))
}

/// Serializes two arrays as separate row groups in an in-memory Parquet file.
///
/// Both arrays use the same field and exercise multi-batch ingestion paths.
///
/// # Examples
/// ```ignore
/// let field = feature_field(2, false, false)?;
/// let first = build_array(&[[1.0, 2.0]])?;
/// let second = build_array(&[[3.0, 4.0]])?;
/// let parquet = write_parquet_two_batches(first, second, field)?;
/// assert!(!parquet.is_empty());
/// # Ok::<(), FixtureError>(())
/// ```
///
/// # Errors
/// Returns [`FixtureError::Arrow`] if either array does not match the schema,
/// or [`FixtureError::Parquet`] if serialization fails.
pub fn write_parquet_two_batches(
    first: FixedSizeListArray,
    second: FixedSizeListArray,
    field: Field,
) -> Result<Bytes, FixtureError> {
    let schema = Arc::new(Schema::new(vec![field]));
    let first_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(first) as ArrayRef])?;
    let second_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(second) as ArrayRef])?;
    let mut buffer = Vec::new();
    write_record_batches(&mut buffer, schema, &[first_batch, second_batch])?;
    Ok(Bytes::from(buffer))
}

/// Serializes a fixed-size list array to the supplied file.
///
/// The same record-batch serializer powers this file output and the in-memory
/// [`Bytes`] helpers above.
///
/// # Examples
/// ```ignore
/// let field = feature_field(2, false, false)?;
/// let array = build_array(&[[1.0, 2.0]])?;
/// let directory = cap_std::fs::Dir::open_ambient_dir(
///     ".",
///     cap_std::ambient_authority(),
/// )?;
/// let mut file = directory.create("fixture.parquet")?;
/// write_parquet_file(&mut file, field, array)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
/// Returns [`FixtureError::Arrow`] if the array does not match the schema, or
/// [`FixtureError::Parquet`] if serialization fails.
pub fn write_parquet_file<W: Write + Send>(
    file: &mut W,
    field: Field,
    array: FixedSizeListArray,
) -> Result<(), FixtureError> {
    let (schema, batch) = record_batch(field, array)?;
    write_record_batches(file, schema, &[batch])
}

/// Checks that one row has the requested width.
const fn ensure_row_len(row: usize, actual: usize, dimension: usize) -> Result<(), FixtureError> {
    if actual == dimension {
        return Ok(());
    }
    Err(FixtureError::RowLength {
        row,
        expected: dimension,
        actual,
    })
}

/// Checks each row, returning the first row-length error.
fn ensure_rows_len<T>(rows: &[Vec<T>], dimension: usize) -> Result<(), FixtureError> {
    for (index, row) in rows.iter().enumerate() {
        ensure_row_len(index, row.len(), dimension)?;
    }
    Ok(())
}

/// Wraps flat values in the requested Arrow fixed-size-list shape.
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

/// Creates one record batch and retains its matching schema.
fn record_batch(
    field: Field,
    array: FixedSizeListArray,
) -> Result<(Arc<Schema>, RecordBatch), FixtureError> {
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array) as ArrayRef])?;
    Ok((schema, batch))
}

/// Serializes record batches through one writer implementation.
fn write_record_batches<W: Write + Send>(
    output: W,
    schema: Arc<Schema>,
    batches: &[RecordBatch],
) -> Result<(), FixtureError> {
    let mut writer = ArrowWriter::try_new(output, schema, None)?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.close()?;
    Ok(())
}

#[cfg(test)]
mod tests;
