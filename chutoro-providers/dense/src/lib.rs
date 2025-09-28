//! Dense providers for f32 vectors backed by contiguous storage.
use std::{fs::File, path::Path};

use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatchReader};
use arrow_schema::{ArrowError, DataType};
use chutoro_core::{DataSource, DataSourceError};
use parquet::arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder};
use parquet::file::reader::ChunkReader;
use thiserror::Error;

/// In-memory dense vector data source.
pub struct DenseSource {
    data: Vec<Vec<f32>>,
    name: String,
}

impl DenseSource {
    /// Creates a new dense source.
    ///
    /// # Panics
    /// Panics if row lengths differ; use [`try_new`] for fallible construction.
    ///
    /// # Examples
    /// ```
    /// use chutoro_providers_dense::DenseSource;
    /// let ds = DenseSource::new("demo", vec![vec![0.0], vec![1.0]]);
    /// assert_eq!(ds.len(), 2);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn new(name: impl Into<String>, data: Vec<Vec<f32>>) -> Self {
        #[expect(
            clippy::expect_used,
            reason = "constructor panics on inconsistent row lengths"
        )]
        Self::try_new(name, data).expect("rows must have equal length")
    }

    /// Creates a dense source after validating uniform dimensions.
    ///
    /// # Errors
    /// Returns `DataSourceError::DimensionMismatch` if row lengths differ.
    ///
    /// # Examples
    /// ```
    /// use chutoro_providers_dense::DenseSource;
    /// use chutoro_core::DataSourceError;
    /// let err = DenseSource::try_new("demo", vec![vec![0.0], vec![1.0, 2.0]]);
    /// assert!(matches!(err, Err(DataSourceError::DimensionMismatch { .. })));
    /// ```
    pub fn try_new(name: impl Into<String>, data: Vec<Vec<f32>>) -> Result<Self, DataSourceError> {
        if let Some((first, rest)) = data.split_first() {
            let dim = first.len();
            for row in rest {
                if row.len() != dim {
                    return Err(DataSourceError::DimensionMismatch {
                        left: dim,
                        right: row.len(),
                    });
                }
            }
        }
        Ok(Self {
            data,
            name: name.into(),
        })
    }
}

impl DataSource for DenseSource {
    fn len(&self) -> usize {
        self.data.len()
    }
    fn name(&self) -> &str {
        &self.name
    }
    #[expect(clippy::float_arithmetic, reason = "vector arithmetic")]
    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let a = self
            .data
            .get(i)
            .ok_or(DataSourceError::OutOfBounds { index: i })?;
        let b = self
            .data
            .get(j)
            .ok_or(DataSourceError::OutOfBounds { index: j })?;
        if a.len() != b.len() {
            return Err(DataSourceError::DimensionMismatch {
                left: a.len(),
                right: b.len(),
            });
        }
        let sum = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum::<f32>();
        Ok(sum.sqrt())
    }
}

/// Dense matrix provider backed by a contiguous row-major buffer.
///
/// # Examples
/// ```
/// use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
/// use chutoro_providers_dense::DenseMatrixProvider;
/// use chutoro_core::DataSource;
///
/// let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), 2);
/// builder.values().append_value(1.0);
/// builder.values().append_value(2.0);
/// builder.append(true);
/// let array = builder.finish();
///
/// let provider = DenseMatrixProvider::try_from_fixed_size_list("demo", &array).unwrap();
/// assert_eq!(provider.len(), 1);
/// assert_eq!(provider.dimension(), 2);
/// assert_eq!(provider.data(), &[1.0, 2.0]);
/// ```
#[derive(Debug)]
pub struct DenseMatrixProvider {
    name: String,
    rows: usize,
    dimension: usize,
    values: Vec<f32>,
}

impl DenseMatrixProvider {
    /// Creates a provider from a contiguous matrix.
    fn from_parts(
        name: impl Into<String>,
        rows: usize,
        dimension: usize,
        values: Vec<f32>,
    ) -> Self {
        Self {
            name: name.into(),
            rows,
            dimension,
            values,
        }
    }

    /// Returns the dimensionality of each row.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the underlying row-major matrix.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.values
    }

    /// Loads data from an Arrow [`FixedSizeListArray`].
    pub fn try_from_fixed_size_list(
        name: impl Into<String>,
        array: &FixedSizeListArray,
    ) -> Result<Self, DenseMatrixProviderError> {
        let dimension = validate_fixed_size_list(array)?;
        let mut values = Vec::with_capacity(array.len() * dimension);
        copy_list_values(array, dimension, 0, &mut values)?;
        Ok(Self::from_parts(name, array.len(), dimension, values))
    }

    /// Loads data from a Parquet column containing `FixedSizeList<Float32, D>` rows.
    pub fn try_from_parquet_path(
        name: impl Into<String>,
        path: impl AsRef<Path>,
        column: &str,
    ) -> Result<Self, DenseMatrixProviderError> {
        let file = File::open(path)?;
        Self::try_from_parquet_reader(name, file, column)
    }

    /// Loads data from a Parquet reader.
    pub fn try_from_parquet_reader<R>(
        name: impl Into<String>,
        reader: R,
        column: &str,
    ) -> Result<Self, DenseMatrixProviderError>
    where
        R: ChunkReader + Send + 'static,
    {
        let builder = ParquetRecordBatchReaderBuilder::try_new(reader)?;
        let mask = ProjectionMask::columns(builder.parquet_schema(), [column]);
        let reader = builder.with_projection(mask).build()?;
        let schema = reader.schema();
        let column_index =
            schema
                .index_of(column)
                .map_err(|_| DenseMatrixProviderError::ColumnNotFound {
                    column: column.to_owned(),
                })?;
        let field = schema.field(column_index);
        let width = match field.data_type() {
            DataType::FixedSizeList(child, width) => {
                if child.data_type() != &DataType::Float32 {
                    return Err(DenseMatrixProviderError::InvalidListValueType {
                        actual: child.data_type().clone(),
                    });
                }
                *width
            }
            other => {
                return Err(DenseMatrixProviderError::InvalidColumnType {
                    column: column.to_owned(),
                    actual: other.clone(),
                });
            }
        };
        let dimension = usize::try_from(width)
            .map_err(|_| DenseMatrixProviderError::InvalidDimension { actual: width })?;
        let mut values = Vec::new();
        let mut rows = 0_usize;
        for batch in reader {
            let batch = batch?;
            let column_array = batch.column(column_index);
            let list = column_array
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| DenseMatrixProviderError::InvalidColumnType {
                    column: column.to_owned(),
                    actual: column_array.data_type().clone(),
                })?;
            let list_width = usize::try_from(list.value_length()).map_err(|_| {
                DenseMatrixProviderError::InvalidDimension {
                    actual: list.value_length(),
                }
            })?;
            if list_width != dimension {
                return Err(DenseMatrixProviderError::InconsistentBatchDimension {
                    expected: dimension,
                    actual: list_width,
                });
            }
            copy_list_values(list, dimension, rows, &mut values)?;
            rows += list.len();
        }
        Ok(Self::from_parts(name, rows, dimension, values))
    }

    fn row_slice(&self, index: usize) -> Result<&[f32], DataSourceError> {
        if index >= self.rows {
            return Err(DataSourceError::OutOfBounds { index });
        }
        let start = index * self.dimension;
        let end = start + self.dimension;
        Ok(&self.values[start..end])
    }
}

impl DataSource for DenseMatrixProvider {
    fn len(&self) -> usize {
        self.rows
    }

    fn name(&self) -> &str {
        &self.name
    }

    #[expect(clippy::float_arithmetic, reason = "vector arithmetic")]
    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let a = self.row_slice(i)?;
        let b = self.row_slice(j)?;
        let mut sum = 0.0_f32;
        for idx in 0..self.dimension {
            let diff = a[idx] - b[idx];
            sum += diff * diff;
        }
        Ok(sum.sqrt())
    }
}

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
    #[error("inconsistent dimensions across batches: expected {expected}, got {actual}")]
    InconsistentBatchDimension { expected: usize, actual: usize },
    #[error("arrow error: {0}")]
    Arrow(#[from] ArrowError),
    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
}

fn validate_fixed_size_list(array: &FixedSizeListArray) -> Result<usize, DenseMatrixProviderError> {
    let value_type = array.value_type();
    if value_type != DataType::Float32 {
        return Err(DenseMatrixProviderError::InvalidListValueType { actual: value_type });
    }
    usize::try_from(array.value_length()).map_err(|_| DenseMatrixProviderError::InvalidDimension {
        actual: array.value_length(),
    })
}

fn copy_list_values(
    array: &FixedSizeListArray,
    dimension: usize,
    start_row: usize,
    out: &mut Vec<f32>,
) -> Result<usize, DenseMatrixProviderError> {
    out.reserve(array.len() * dimension);
    for row_index in 0..array.len() {
        let absolute_row = start_row + row_index;
        if array.is_null(row_index) {
            return Err(DenseMatrixProviderError::NullRow { row: absolute_row });
        }
        let row = array.value(row_index);
        let floats = row.as_any().downcast_ref::<Float32Array>().ok_or_else(|| {
            DenseMatrixProviderError::InvalidListValueType {
                actual: row.data_type().clone(),
            }
        })?;
        if floats.len() != dimension {
            return Err(DenseMatrixProviderError::InvalidRowLength {
                row: absolute_row,
                expected: dimension,
                actual: floats.len(),
            });
        }
        if floats.null_count() > 0 {
            let value_index = (0..dimension)
                .find(|&idx| floats.is_null(idx))
                .unwrap_or_default();
            return Err(DenseMatrixProviderError::NullValue {
                row: absolute_row,
                value_index,
            });
        }
        for idx in 0..dimension {
            out.push(floats.value(idx));
        }
    }
    Ok(array.len())
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests require contextual panics")]
mod tests {
    use super::*;
    use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
    use arrow_array::{ArrayRef, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use bytes::Bytes;
    use parquet::arrow::arrow_writer::ArrowWriter;
    use rstest::rstest;
    use std::sync::Arc;

    #[rstest]
    fn distance_dimension_mismatch() {
        let ds = DenseSource {
            data: vec![vec![0.0], vec![1.0, 2.0]],
            name: "d".into(),
        };
        let err = ds
            .distance(0, 1)
            .expect_err("distance must validate dimensions");
        assert!(matches!(err, DataSourceError::DimensionMismatch { .. }));
    }

    #[rstest]
    fn try_new_rejects_mismatched_rows() {
        let err = DenseSource::try_new("d", vec![vec![0.0], vec![1.0, 2.0]]);
        assert!(matches!(
            err,
            Err(DataSourceError::DimensionMismatch { .. })
        ));
    }
    #[rstest]
    fn distance_out_of_bounds() {
        let ds = DenseSource::try_new("d", vec![vec![0.0], vec![1.0]]).expect("rows must match");
        let err = ds
            .distance(0, 99)
            .expect_err("distance must report out-of-bounds");
        assert!(matches!(err, DataSourceError::OutOfBounds { index: 99 }));
    }

    #[rstest]
    fn distance_ok() {
        let ds = DenseSource::try_new("d", vec![vec![0.0, 0.0], vec![3.0, 4.0]])
            .expect("valid uniform rows");
        let d = ds.distance(0, 1).expect("distance must succeed");
        assert!((d - 5.0).abs() < 1e-6);
    }

    fn build_array(rows: &[[f32; 3]]) -> FixedSizeListArray {
        let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), 3);
        for row in rows {
            for value in row {
                builder.values().append_value(*value);
            }
            builder.append(true);
        }
        builder.finish()
    }

    #[rstest]
    fn matrix_provider_from_fixed_size_list() {
        let array = build_array(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let provider =
            DenseMatrixProvider::try_from_fixed_size_list("demo", &array).expect("valid matrix");
        assert_eq!(provider.len(), 2);
        assert_eq!(provider.dimension(), 3);
        assert_eq!(provider.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let distance = provider.distance(0, 1).expect("distance should work");
        assert!((distance - (27.0_f32).sqrt()).abs() < 1e-6);
    }

    #[rstest]
    fn matrix_provider_rejects_null_rows() {
        let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), 2);
        builder.values().append_value(1.0);
        builder.values().append_value(2.0);
        builder.append(true);
        builder.values().append_null();
        builder.values().append_null();
        builder.append(false);
        let array = builder.finish();
        let err = DenseMatrixProvider::try_from_fixed_size_list("demo", &array)
            .expect_err("null rows must be rejected");
        assert!(matches!(err, DenseMatrixProviderError::NullRow { row: 1 }));
    }

    #[rstest]
    fn matrix_provider_rejects_null_values() {
        let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), 2);
        builder.values().append_value(1.0);
        builder.values().append_value(2.0);
        builder.append(true);
        builder.values().append_value(3.0);
        builder.values().append_null();
        builder.append(true);
        let array = builder.finish();
        let err = DenseMatrixProvider::try_from_fixed_size_list("demo", &array)
            .expect_err("null values must be rejected");
        assert!(matches!(
            err,
            DenseMatrixProviderError::NullValue {
                row: 1,
                value_index: 1
            }
        ));
    }

    #[rstest]
    fn matrix_provider_rejects_non_float_children() {
        let field = Arc::new(Field::new("item", DataType::Int32, true));
        let values: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![1, 2, 3, 4]));
        let array = FixedSizeListArray::new(field, 2, values, None);
        let err = DenseMatrixProvider::try_from_fixed_size_list("demo", &array)
            .expect_err("non-float children must be rejected");
        assert!(matches!(
            err,
            DenseMatrixProviderError::InvalidListValueType { .. }
        ));
    }

    fn write_parquet(array: FixedSizeListArray) -> Bytes {
        let field = Field::new(
            "features",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3),
            false,
        );
        let schema = Arc::new(Schema::new(vec![field]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(array) as ArrayRef]).expect("batch");
        let mut buffer = Vec::new();
        {
            let mut writer = ArrowWriter::try_new(&mut buffer, schema, None).expect("writer");
            writer.write(&batch).expect("write");
            writer.close().expect("close");
        }
        Bytes::from(buffer)
    }

    #[rstest]
    fn matrix_provider_from_parquet() {
        let array = build_array(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let bytes = write_parquet(array);
        let provider = DenseMatrixProvider::try_from_parquet_reader("demo", bytes, "features")
            .expect("parquet load");
        assert_eq!(provider.len(), 2);
        assert_eq!(provider.dimension(), 3);
        assert_eq!(provider.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[rstest]
    fn matrix_provider_parquet_missing_column() {
        let array = build_array(&[[1.0, 2.0, 3.0]]);
        let bytes = write_parquet(array);
        let err = DenseMatrixProvider::try_from_parquet_reader("demo", bytes, "unknown")
            .expect_err("missing column");
        assert!(matches!(
            err,
            DenseMatrixProviderError::ColumnNotFound { column } if column == "unknown"
        ));
    }

    #[rstest]
    fn matrix_provider_parquet_wrong_type() {
        let field = Field::new("features", DataType::Int32, false);
        let schema = Arc::new(Schema::new(vec![field]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(arrow_array::Int32Array::from(vec![1, 2, 3])) as ArrayRef],
        )
        .expect("batch");
        let mut buffer = Vec::new();
        {
            let mut writer = ArrowWriter::try_new(&mut buffer, schema, None).expect("writer");
            writer.write(&batch).expect("write");
            writer.close().expect("close");
        }
        let bytes = Bytes::from(buffer);
        let err = DenseMatrixProvider::try_from_parquet_reader("demo", bytes, "features")
            .expect_err("wrong type");
        assert!(matches!(
            err,
            DenseMatrixProviderError::InvalidColumnType { .. }
        ));
    }
}
