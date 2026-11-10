//! Dense-provider-specific helpers for record-batch ingestion tests.

use super::{DenseMatrixProvider, DenseMatrixProviderError};
use crate::ingest::{append_fixed_size_list_values, validate_fixed_size_list_field};
use arrow_array::{Array, FixedSizeListArray, RecordBatch};

/// Builds a dense provider from record batches without using the Parquet reader.
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
