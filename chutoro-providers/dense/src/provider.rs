//! Dense matrix provider implementation and ingestion utilities.
use std::{fs::File, path::Path};

use arrow_array::{Array, FixedSizeListArray, RecordBatchReader};

use chutoro_core::{DataSource, DataSourceError};
use parquet::arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder};
use parquet::file::reader::ChunkReader;

use crate::errors::DenseMatrixProviderError;
use crate::ingest::{append_fixed_size_list_values, validate_fixed_size_list_field};

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
    pub(crate) fn from_parts(
        name: impl Into<String>,
        rows: usize,
        dimension: usize,
        values: Vec<f32>,
    ) -> Self {
        debug_assert_eq!(values.len(), rows.saturating_mul(dimension));
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
        let start = index
            .checked_mul(self.dimension)
            .ok_or(DataSourceError::OutOfBounds { index })?;
        let end = start
            .checked_add(self.dimension)
            .ok_or(DataSourceError::OutOfBounds { index })?;
        if end > self.values.len() {
            return Err(DataSourceError::OutOfBounds { index });
        }
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

    #[expect(clippy::float_arithmetic, reason = "vector arithmetic")]
    fn distance_batch(
        &self,
        pairs: &[(usize, usize)],
        out: &mut [f32],
    ) -> Result<(), DataSourceError> {
        if pairs.len() != out.len() {
            return Err(DataSourceError::OutputLengthMismatch {
                out: out.len(),
                expected: pairs.len(),
            });
        }
        let dimension = self.dimension;
        for (idx, &(left, right)) in pairs.iter().enumerate() {
            let a = self.row_slice(left)?;
            let b = self.row_slice(right)?;
            let mut sum = 0.0_f32;
            for value_index in 0..dimension {
                let diff = a[value_index] - b[value_index];
                sum += diff * diff;
            }
            out[idx] = sum.sqrt();
        }
        Ok(())
    }
}
