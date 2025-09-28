use std::{fs::File, path::Path};

use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatchReader};
use arrow_schema::{DataType, Field};
use chutoro_core::{DataSource, DataSourceError};
use parquet::arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder};
use parquet::file::reader::ChunkReader;

use crate::errors::DenseMatrixProviderError;

/// Dense matrix provider backed by a contiguous row-major buffer.
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
        let mut values = Vec::new();
        let dimension = append_fixed_size_list_values(array, None, 0, &mut values)?;
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
        let dimension = validate_fixed_size_list_field(field, column)?;
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
            append_fixed_size_list_values(list, Some(dimension), rows, &mut values)?;
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

fn validate_fixed_size_list_field(
    field: &Field,
    column: &str,
) -> Result<usize, DenseMatrixProviderError> {
    match field.data_type() {
        DataType::FixedSizeList(child, width) => {
            if child.data_type() != &DataType::Float32 {
                return Err(DenseMatrixProviderError::InvalidListValueType {
                    actual: child.data_type().clone(),
                });
            }
            usize::try_from(*width)
                .map_err(|_| DenseMatrixProviderError::InvalidDimension { actual: *width })
        }
        other => Err(DenseMatrixProviderError::InvalidColumnType {
            column: column.to_owned(),
            actual: other.clone(),
        }),
    }
}

fn append_fixed_size_list_values(
    array: &FixedSizeListArray,
    expected_dimension: Option<usize>,
    start_row: usize,
    out: &mut Vec<f32>,
) -> Result<usize, DenseMatrixProviderError> {
    let dimension = validate_fixed_size_list(array)?;
    if let Some(expected) = expected_dimension.filter(|&expected| expected != dimension) {
        return Err(DenseMatrixProviderError::InconsistentBatchDimension {
            expected,
            actual: dimension,
        });
    }
    copy_list_values(array, dimension, start_row, out)?;
    Ok(dimension)
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
) -> Result<(), DenseMatrixProviderError> {
    let rows = array.len();
    let additional = rows
        .checked_mul(dimension)
        .ok_or(DenseMatrixProviderError::CapacityOverflow { rows, dimension })?;
    out.reserve(additional);
    for row_index in 0..rows {
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
        let values = floats.values().as_ref();
        let start = floats.offset();
        let end = start + dimension;
        out.extend_from_slice(&values[start..end]);
    }
    Ok(())
}
