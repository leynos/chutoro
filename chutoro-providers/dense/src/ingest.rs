//! Helpers for ingesting fixed-size list arrays into dense buffers.
use arrow_array::{Array, FixedSizeListArray, Float32Array};
use arrow_schema::{DataType, Field};

use crate::errors::DenseMatrixProviderError;

pub(crate) fn validate_fixed_size_list_field(
    field: &Field,
    column: &str,
) -> Result<usize, DenseMatrixProviderError> {
    match field.data_type() {
        DataType::FixedSizeList(child, width) => {
            if field.is_nullable() || child.is_nullable() {
                return Err(DenseMatrixProviderError::NullableField {
                    column: column.to_owned(),
                    nullable_child: child.is_nullable(),
                });
            }
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

pub(crate) fn append_fixed_size_list_values(
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

pub(crate) fn validate_fixed_size_list(
    array: &FixedSizeListArray,
) -> Result<usize, DenseMatrixProviderError> {
    let value_type = array.value_type();
    if value_type != DataType::Float32 {
        return Err(DenseMatrixProviderError::InvalidListValueType { actual: value_type });
    }
    usize::try_from(array.value_length()).map_err(|_| DenseMatrixProviderError::InvalidDimension {
        actual: array.value_length(),
    })
}

pub(crate) fn copy_list_values(
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
                .expect("null_count > 0 but no null index found");
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
